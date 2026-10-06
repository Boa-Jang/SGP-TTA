"""Source vs SGP-TTA inference on the bundled sample images.

Runs the source model and the SGP-TTA-wrapped model on every image in
``sample/`` and saves a 5-panel comparison figure per image under
``outputs/``. Mirrors ``notebook/examples.ipynb`` as a CLI entry point.

Usage
-----
    # from repo root
    python script/inference.py
    python script/inference.py --checkpoint model/DRIVE_unet.pth --tau 5.0

Prerequisites
-------------
    1. ``pip install -r requirements.txt``
    2. Place the pre-trained checkpoint at ``model/DRIVE_unet.pth``
       (see README for the Google Drive link).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


# Resolve the repo root (the directory that contains ``sgp_tta.py``) by
# walking up from this file. Lets the script be invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sgp_tta import SGPTTA  # noqa: E402


# =============================================================================
# Data
# =============================================================================
def should_invert(path, invert_default):
    """Per-image invert decision.

    The DRIVE source checkpoint sees dark vessels on a bright fundus
    background. STARE samples share that polarity and must NOT be
    inverted; OCTA / road-style targets have the opposite polarity
    (bright structure on dark background) and benefit from the 1 - x
    inversion.
    """
    if Path(path).stem.lower().startswith('stare'):
        return False
    return invert_default


class InferenceDataset(Dataset):
    def __init__(self, image_paths, image_size, invert_default=True):
        self.image_paths = list(image_paths)
        self.invert_default = invert_default
        self.transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image_rgb = np.array(Image.open(path).convert('RGB'))
        image_tensor = self.transform(image=image_rgb)['image']
        if should_invert(path, self.invert_default):
            image_tensor = 1.0 - image_tensor
        return {
            'image':     image_tensor,
            'image_rgb': image_rgb,
            'path':      str(path),
        }


# =============================================================================
# Model
# =============================================================================
def build_model(checkpoint_path, encoder, in_channels, num_classes, device):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def predict_source(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        logits = model(batch['image'].to(device))
        probs.append(torch.sigmoid(logits)[0, 0].cpu().numpy())
    return probs


def predict_sgptta(model, loader, device, args):
    tta = SGPTTA(
        model=model,
        temperature=args.tau,
        lr=args.lr,
        num_steps=args.num_steps,
        entropy_weight=args.entropy_weight,
        skeleton_weight=args.skeleton_weight,
        dilate_radius=args.dilate_radius,
    )
    probs = []
    for batch in loader:
        logits = tta(batch['image'].to(device))
        probs.append(torch.sigmoid(logits)[0, 0].detach().cpu().numpy())
    return probs, tta.get_stats()


# =============================================================================
# Visualization
# =============================================================================
def save_panel(image_rgb, src_prob, tta_prob, out_path, threshold=0.5):
    src_mask = (src_prob > threshold).astype(np.uint8)
    tta_mask = (tta_prob > threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(image_rgb);                             axes[0].set_title('Input')
    axes[1].imshow(src_prob, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Source | prob.')
    axes[2].imshow(src_mask, cmap='gray', vmin=0, vmax=1); axes[2].set_title('Source | mask')
    axes[3].imshow(tta_prob, cmap='gray', vmin=0, vmax=1); axes[3].set_title('SGP-TTA | prob.')
    axes[4].imshow(tta_mask, cmap='gray', vmin=0, vmax=1); axes[4].set_title('SGP-TTA | mask')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Source vs SGP-TTA inference on sample images.'
    )
    p.add_argument('--checkpoint', type=Path, default=REPO_ROOT / 'model' / 'DRIVE_unet.pth')
    p.add_argument('--sample-dir', type=Path, default=REPO_ROOT / 'sample')
    p.add_argument('--out-dir',    type=Path, default=REPO_ROOT / 'outputs')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--encoder',    type=str, default='resnet50')
    p.add_argument(
        '--no-invert', action='store_true',
        help='Disable the default 1 - x RGB inversion. Note that STARE '
             'samples are already forced to invert=False regardless of '
             'this flag (see should_invert).'
    )
    # SGP-TTA hyperparameters
    p.add_argument('--tau',             type=float, default=5.0, help='ProgBN warmup temperature.')
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--num-steps',       type=int,   default=1)
    p.add_argument('--entropy-weight',  type=float, default=1.0)
    p.add_argument('--skeleton-weight', type=float, default=1.0)
    p.add_argument('--dilate-radius',   type=int,   default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    assert args.checkpoint.exists(), f'Checkpoint not found: {args.checkpoint}'
    image_paths = sorted(args.sample_dir.glob('*.png'))
    assert image_paths, f'No *.png images found in {args.sample_dir}'

    invert_default = not args.no_invert
    dataset = InferenceDataset(image_paths, args.image_size, invert_default=invert_default)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f'{len(dataset)} samples | image_size={args.image_size} | invert_default={invert_default}')
    for p in image_paths:
        print(f'  {Path(p).name:<20s} invert={should_invert(p, invert_default)}')

    print('-> Source-only inference')
    src_model = build_model(args.checkpoint, args.encoder, 3, 1, device)
    src_probs = predict_source(src_model, loader, device)

    print('-> SGP-TTA adaptation')
    tta_model = build_model(args.checkpoint, args.encoder, 3, 1, device)
    tta_probs, stats = predict_sgptta(tta_model, loader, device, args)
    print(f'   stats: {stats}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(dataset):
        name = Path(batch['path']).stem
        out  = args.out_dir / f'{name}.png'
        save_panel(batch['image_rgb'], src_probs[i], tta_probs[i], out)
        print(f'   saved {out}')


if __name__ == '__main__':
    main()
