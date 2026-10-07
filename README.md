# SGP-TTA

**Skeleton-Guided Progressive Test-Time Adaptation for Thin Curvilinear Structures**

<div align="center">
  <img width="100%" alt="SGP-TTA Overview" src="overview.png">
</div>

SGP-TTA is a single-image test-time adaptation (TTA) wrapper for
segmentation networks that predict *thin curvilinear* structures —
retinal vessels across modalities (color fundus, FA, OCTA), roads in
aerial imagery, and similar. It adapts a source-domain model to each
incoming target-domain image without any labels and without touching
the training pipeline.

**Project page:** https://boa-jang.github.io/SGPTTA/

## Method

The method has two ingredients:

1. **Progressive Batch Normalization (ProgBN).** Every `nn.BatchNorm2d` layer
   in the source model is replaced by a layer that blends the frozen source
   running statistics with the current-batch target statistics using a
   monotonic square-root warmup:

   $$\alpha_n = \frac{\sqrt{n}}{\sqrt{n} + \tau}, \qquad \mu = \alpha_n\, \mu_{\text{target}} + (1 - \alpha_n)\, \mu_{\text{source}}$$

   Small `tau` warms up quickly; large `tau` stays anchored to source
   statistics for longer.

2. **Skeleton-Guided objective.** For each incoming image we form six
   augmented views (identity, two flips, three 90-degree rotations), average
   their per-view predictions into a **Multi-View Consensus (MVC)** map, and
   update only the BN affine parameters (`gamma`, `beta`) with two losses:

   - **Consensus entropy** on the mean prediction.
   - **Consensus Skeleton Recall (CSR) loss** — skeletonize the binarized
     consensus, dilate it into a thin tube, and penalize `1 - recall` of each
     view's probability against that tube. This anchors the network to the
     structural centerline and keeps thin branches from being erased by the
     entropy pressure toward confident predictions.

## Environment

Requires Python 3.9+. Install dependencies with:

```bash
pip install -r requirements.txt
```

Tested with:
```
Python 3.9
PyTorch 2.6.0 + CUDA 12.2
segmentation-models-pytorch 0.3
```

## Pre-trained models

Source-domain checkpoints are too large to bundle with the repository.
Download them from Google Drive, create a `model/` folder at the repository
root, and place the `.pth` file(s) inside:

- [Google Drive — SGP-TTA pre-trained models](https://drive.google.com/drive/folders/1QS-IQ3GwHPEJgSvr_lJzYCkiFbkuqFcn?usp=drive_link)

```bash
mkdir -p model
# move the downloaded DRIVE_unet.pth into ./model/
```

Expected layout after download:
```
model/
└── DRIVE_unet.pth   # smp.Unet (resnet50 encoder), trained on DRIVE retinal vessels
```

## Quick start

Once the checkpoint is in `model/`, there are two equivalent entry
points — both load `model/DRIVE_unet.pth`, run a **Source-only**
prediction on each sample in `sample/`, then run **SGP-TTA** on the same
stream and compare the two predictions side by side.

**Notebook** — exploratory, inline figures:

```bash
jupyter lab notebook/examples.ipynb
```

Run top to bottom. The notebook auto-detects the repo root, so it works
whether you launch Jupyter from the repository root or from `notebook/`.
Edit the config cell to point to a different checkpoint or sample
directory.

**Command-line script** — one shot, writes figures to disk:

```bash
# from the repository root
python script/inference.py
```

By default the script loads `model/DRIVE_unet.pth`, runs on every PNG in
`sample/`, and saves a 5-panel comparison figure (`Input | Source prob. |
Source mask | SGP-TTA prob. | SGP-TTA mask`) per image under
`outputs/`.

Common overrides:

```bash
# different checkpoint / sample directory / output directory
python script/inference.py \
    --checkpoint model/other.pth \
    --sample-dir path/to/images \
    --out-dir    runs/other

# tune the ProgBN warmup and SGP-TTA step
python script/inference.py --tau 1.0 --num-steps 2 --lr 5e-4

# disable the 1 - x RGB inversion
python script/inference.py --no-invert
```

Run `python script/inference.py --help` for the full flag list.

## Usage

`SGPTTA` is model-agnostic — it works on any segmentation network that emits
raw logits (or a tuple whose first element is logits) and contains
`nn.BatchNorm2d` layers.

```python
import torch
from sgp_tta import SGPTTA

model = build_your_segmentation_network()          # e.g. U-Net with BN
model.load_state_dict(torch.load('source.pth'))
model = model.to('cuda').eval()

tta = SGPTTA(
    model,
    temperature=5.0,       # ProgBN warmup tau
    lr=1e-3,               # Adam LR for BN affine params
    num_steps=1,           # gradient steps per image
    entropy_weight=1.0,    # weight of consensus entropy loss
    skeleton_weight=1.0,   # weight of consensus skeleton recall loss
    dilate_radius=2,       # disk radius for skeleton dilation
)

for image in target_stream:                        # shape [1, C, H, W]
    logits = tta(image.to('cuda'))                 # adapted logits
    pred = (torch.sigmoid(logits) > 0.5).float()

# Call reset() before adapting to a new target domain:
tta.reset()
```

### What gets updated

- **Trainable:** BN affine parameters (`gamma`, `beta`) only.
- **Frozen:** all conv/linear weights, all bias terms outside BN, both
  running statistics buffers.
- **Blended per forward:** the batch mean/variance used inside each ProgBN
  layer, following the warmup schedule.

### Notes

- The wrapper modifies `model` **in place** (its `BatchNorm2d` layers are
  swapped for `ProgBN`). If you need to keep an untouched copy of the source
  model, deep-copy it before wrapping.
- `SGPTTA.__call__` returns adapted **logits**, so downstream code can keep
  applying `torch.sigmoid` and thresholding.
- Small `temperature` (e.g. `1`) warms up quickly and is good for streams
  where the target domain is well-represented after only a handful of
  images. Larger `temperature` (e.g. `5`) stays anchored to source
  statistics for longer, which helps on noisier or more shifted targets.

## Repository layout

```
SGPTTA/
├── sgp_tta.py              # ProgBN + SGPTTA (the whole method)
├── script/
│   └── inference.py        # CLI: Source vs SGP-TTA on sample/, saves panels
├── notebook/
│   └── examples.ipynb      # same demo as a notebook (inline figures)
├── model/                  # downloaded checkpoints go here (gitignored)
│   └── DRIVE_unet.pth
├── sample/                 # example target-domain images
├── overview.png
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If this code is helpful for your research, please cite:

```bibtex
@article{jang2026sgptta,
  title   = {Skeleton-Guided Progressive Test-Time Adaptation for Thin Curvilinear Structures},
  author  = {Jang, Boa and Lee, JunGyu and Lee, Gwanho and Choi, Jinwook and Kim, Young-Gon},
  year    = {2026},
  note    = {Under review}
}
```

## License

See [`LICENSE`](LICENSE).
