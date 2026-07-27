"""
SGP-TTA: Skeleton-Guided Progressive Test-Time Adaptation
=========================================================

A single-image test-time adaptation wrapper for tubular-structure
segmentation (retinal vessels, OCTA vasculature, roads, etc.).

The method has two components:

1. Progressive Batch Normalization (ProgBN):
   Replaces every ``nn.BatchNorm2d`` in the source model with a layer that
   blends the frozen source running statistics with the current-batch
   statistics of the target image, using a monotonic square-root schedule
   of the target-sample counter ``n``:

       alpha_n = sqrt(n) / (sqrt(n) + tau)
       mu_new  = alpha_n * mu_target + (1 - alpha_n) * mu_source
       var_new = alpha_n * var_target + (1 - alpha_n) * var_source

   Small ``tau`` warms up quickly, large ``tau`` stays close to source
   statistics for longer.

2. Skeleton-Guided objective:
   For each incoming image we form six augmented views (identity, two
   flips, three 90-degree rotations), compute their per-view sigmoid
   predictions, average them into a consensus probability map, and
   optimize the BN affine parameters (gamma, beta) with two losses:

       L_entropy: binary entropy of the consensus map
                  H(p) = -p log p - (1-p) log(1-p)

       L_CSR:     consensus skeleton recall
                  1. skeletonize (mean_pred > 0.5) and dilate the skeleton
                     by a disk of radius ``dilate_radius``
                  2. for every view, compute recall of that view's
                     probability against the dilated consensus skeleton
                  3. return the mean of (1 - recall) across views

   The CSR loss anchors the network to the structural centerline of the
   consensus prediction, keeping thin branches from being erased by the
   entropy pressure toward confident predictions.

Only the BN affine parameters (gamma, beta) are updated during
adaptation. All other weights and both running statistics buffers stay
frozen at their source values.
"""

import copy

import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import skeletonize, dilation, disk


# =============================================================================
# Progressive Batch Normalization
# =============================================================================
class ProgBN(nn.BatchNorm2d):
    """Batch-normalization layer that progressively mixes source running
    statistics with the current-batch target statistics.

    The mixing weight follows a square-root warmup schedule of the
    target-sample counter ``sample_num``, controlled by ``temperature``:
    small temperature warms up quickly, large temperature stays anchored
    to source statistics for longer.
    """

    def __init__(self, num_features, temperature=1):
        super().__init__(num_features)
        self.temperature = temperature
        self.sample_num = 0

    def get_mu_var(self, x):
        C = x.shape[1]

        # Current-batch statistics (detached; target signal, no gradient).
        cur_mu = x.mean((0, 2, 3), keepdim=True).detach()
        cur_var = x.var((0, 2, 3), correction=0, keepdim=True).detach()

        # Source statistics from pre-training.
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        # Blend source and target statistics with a monotonic square-root
        # schedule of the target-sample counter.
        alpha = np.sqrt(self.sample_num) / (np.sqrt(self.sample_num) + self.temperature)
        new_mu = alpha * cur_mu + (1 - alpha) * src_mu
        new_var = alpha * cur_var + (1 - alpha) * src_var

        return new_mu, new_var

    def forward(self, x):
        _, C, _, _ = x.shape

        new_mu, new_var = self.get_mu_var(x)

        new_sig = (new_var + self.eps).sqrt()
        out = (
            ((x - new_mu) / new_sig)
            * self.weight.view(1, C, 1, 1)
            + self.bias.view(1, C, 1, 1)
        )
        return out


def convert_bn_to_progbn(model, temperature=1):
    """Recursively replace every ``nn.BatchNorm2d`` in ``model`` with a
    ``ProgBN`` initialized from the original BN's weights, biases, and
    running statistics. Modifies ``model`` in place and returns it.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d) and not isinstance(module, ProgBN):
            progbn = ProgBN(module.num_features, temperature=temperature)
            progbn.load_state_dict(module.state_dict(), strict=False)

            if module.running_mean is not None:
                device = module.running_mean.device
            elif module.weight is not None:
                device = module.weight.device
            else:
                device = torch.device('cpu')

            progbn = progbn.to(device)
            setattr(model, name, progbn)
        else:
            convert_bn_to_progbn(module, temperature=temperature)
    return model


# =============================================================================
# SGP-TTA: Skeleton-Guided Progressive Test-Time Adaptation
# =============================================================================
class SGPTTA:
    """Wraps a trained segmentation model with skeleton-guided,
    progressive-BN test-time adaptation.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation network whose ``nn.BatchNorm2d`` layers
        will be replaced by ``ProgBN`` in place. The network should
        return raw logits (or a tuple whose first element is logits).
    temperature : float, default 1
        Warmup temperature ``tau`` for the ProgBN blending schedule.
    lr : float, default 1e-3
        Learning rate for the Adam optimizer over BN affine parameters.
    num_steps : int, default 1
        Gradient steps per incoming image.
    entropy_weight : float, default 1.0
        Weight of the consensus entropy loss.
    skeleton_weight : float, default 1.0
        Weight of the consensus skeleton recall (CSR) loss.
    dilate_radius : int, default 2
        Radius of the disk used to dilate the consensus skeleton before
        computing recall. Larger radius tolerates small mis-alignments
        between augmented predictions.

    Notes
    -----
    Call ``__call__(x)`` to obtain adapted logits (shape ``[B, 1, H, W]``).
    Call ``reset()`` between different target domains to restore the
    source-model state.
    """

    def __init__(
        self,
        model,
        temperature=1,
        lr=1e-3,
        num_steps=1,
        entropy_weight=1.0,
        skeleton_weight=1.0,
        dilate_radius=2,
    ):
        self.model = model
        self.temperature = temperature
        self.lr = lr
        self.num_steps = num_steps
        self.entropy_weight = entropy_weight
        self.skeleton_weight = skeleton_weight
        self.dilate_radius = dilate_radius

        # Convert BatchNorm2d -> ProgBN in place.
        self.model = convert_bn_to_progbn(model, temperature=temperature)

        # Freeze everything, then unfreeze BN affine params only.
        for param in self.model.parameters():
            param.requires_grad = False

        self.bn_params = []
        for m in self.model.modules():
            if isinstance(m, ProgBN):
                m.weight.requires_grad = True   # gamma
                m.bias.requires_grad = True     # beta
                self.bn_params.append(m.weight)
                self.bn_params.append(m.bias)

        self.optimizer = torch.optim.Adam(self.bn_params, lr=lr)
        self.adaptation_count = 0

        # Snapshot the post-conversion source state so reset() can restore
        # affine params, running stats, and all buffers.
        self.source_state = copy.deepcopy(self.model.state_dict())

    def __call__(self, x):
        """Adapt on ``x`` and return adapted logits."""
        final_prob = self.adapt(x)

        eps = 1e-7
        final_prob = torch.clamp(final_prob, eps, 1 - eps)
        return torch.log(final_prob / (1 - final_prob))

    def _advance_sample_counter(self, batch_size):
        """Advance the target-sample counter once per ``adapt`` call so
        ``sample_num`` matches the number of target images processed
        rather than the number of forward passes.
        """
        self.adaptation_count += batch_size
        for m in self.model.modules():
            if isinstance(m, ProgBN):
                m.sample_num = self.adaptation_count

    def adapt(self, x):
        """Run ``num_steps`` gradient updates on the BN affine parameters
        and return the final adapted probability map (multi-view mean).
        """
        self.model.eval()

        # Advance n once per incoming batch; adaptation and final
        # prediction share the same n from this point on.
        self._advance_sample_counter(batch_size=x.shape[0])

        for _ in range(self.num_steps):
            self.optimizer.zero_grad()

            aug_preds = self._get_augmented_predictions(x)
            mean_pred = aug_preds.mean(dim=0)

            L_entropy = self._compute_entropy_loss(mean_pred)
            L_skeleton = self._consensus_skeleton_recall_loss(aug_preds)

            loss = (
                self.entropy_weight * L_entropy
                + self.skeleton_weight * L_skeleton
            )

            loss.backward()
            self.optimizer.step()

        # Final prediction: multi-view average with the updated affine.
        with torch.no_grad():
            aug_preds = self._get_augmented_predictions(x)
            final_prob = aug_preds.mean(dim=0)

        return final_prob

    def reset(self):
        """Restore the source-model state and rebuild the optimizer so
        adaptation state does not leak across target domains.
        """
        # ``bn_params`` reference the same Parameter objects; load_state_dict
        # writes into their .data in place, so the reference list stays valid.
        self.model.load_state_dict(self.source_state, strict=True)
        self.optimizer = torch.optim.Adam(self.bn_params, lr=self.lr)

        self.adaptation_count = 0
        for m in self.model.modules():
            if isinstance(m, ProgBN):
                m.sample_num = 0

    def get_stats(self):
        return {
            'adaptation_count': self.adaptation_count,
            'num_bn_params': len(self.bn_params),
            'num_steps': self.num_steps,
        }

    def _get_augmented_predictions(self, x):
        """Six-view augmentation: identity, horizontal flip, vertical flip,
        and 90/180/270-degree rotations. Returns a tensor of shape
        ``[6, B, C, H, W]`` in the original spatial frame.
        """
        augmentations = [
            (x, lambda y: y),
            (torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),
            (torch.flip(x, dims=[-2]), lambda y: torch.flip(y, dims=[-2])),
            (torch.rot90(x, k=1, dims=[-2, -1]),
             lambda y: torch.rot90(y, k=3, dims=[-2, -1])),
            (torch.rot90(x, k=2, dims=[-2, -1]),
             lambda y: torch.rot90(y, k=2, dims=[-2, -1])),
            (torch.rot90(x, k=3, dims=[-2, -1]),
             lambda y: torch.rot90(y, k=1, dims=[-2, -1])),
        ]

        predictions = []
        for x_aug, reverse_fn in augmentations:
            output = self.model(x_aug)
            if isinstance(output, tuple):
                output = output[0]

            prob = torch.sigmoid(output)
            predictions.append(reverse_fn(prob))

        return torch.stack(predictions)

    def _compute_entropy_loss(self, pred):
        """Binary entropy of the consensus probability map."""
        eps = 1e-8
        entropy = -(
            pred * torch.log(pred + eps)
            + (1 - pred) * torch.log(1 - pred + eps)
        )
        return entropy.mean()

    def _extract_skeleton(self, pred_binary):
        """Skeletonize each channel of a binary prediction and dilate the
        skeleton by a disk of radius ``self.dilate_radius``. Runs on CPU
        via scikit-image; caller is expected to detach.
        """
        device = pred_binary.device
        pred_np = pred_binary.detach().cpu().numpy()
        B, C, _, _ = pred_np.shape

        skeletons = np.zeros_like(pred_np, dtype=np.float32)
        selem = disk(self.dilate_radius)

        for b in range(B):
            for c in range(C):
                binary = pred_np[b, c] > 0.5
                if binary.sum() > 0:
                    skel = skeletonize(binary)
                    tubed = dilation(skel, selem)
                    skeletons[b, c] = tubed.astype(np.float32)

        return torch.from_numpy(skeletons).to(device)

    def _consensus_skeleton_recall_loss(self, aug_preds):
        """Consensus Skeleton Recall (CSR) loss.

        Binarize the mean prediction, extract and dilate its skeleton,
        then penalize each view by (1 - recall) of that view's
        probability against the dilated consensus skeleton.
        """
        n_aug = aug_preds.shape[0]
        mean_pred = aug_preds.mean(dim=0)

        with torch.no_grad():
            mean_binary = (mean_pred > 0.5).float()
            consensus_skeleton = self._extract_skeleton(mean_binary)

        skeleton_sum = consensus_skeleton.sum() + 1e-8

        total_recall_loss = 0
        for i in range(n_aug):
            recall = (aug_preds[i] * consensus_skeleton).sum() / skeleton_sum
            total_recall_loss += (1 - recall)

        return total_recall_loss / n_aug
