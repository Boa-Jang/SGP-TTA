# SGP-TTA Project Page

Static, single-file project page for SGP-TTA. No build step; everything
loads from CDNs. Open `index.html` locally to preview.

```bash
# from the repository root
cd docs
python -m http.server 8000
# then visit http://localhost:8000
```

## Layout

```
docs/
├── index.html            # the whole page
├── static/
│   ├── css/style.css     # minor overrides on top of Bulma
│   └── images/
│       ├── overview.png          # fig2: method overview (teaser)
│       ├── datasets.png          # fig1: datasets + t-SNE
│       ├── qualitative.png       # fig3: qualitative comparison grid
│       ├── alpha_schedule.png    # fig4: alpha schedule effect
│       ├── architectures.png     # fig5: backbone robustness (radar)
│       └── cost.png              # fig6: clDice vs adaptation cost
└── README.md             # this file
```

## Preparing the figures

The page currently references six PNGs under `static/images/`. Only
`overview.png` is in place; the other five are produced from the paper's
PDF figures. The expected mapping is:

| Paper figure | Save as                              |
| ------------ | ------------------------------------ |
| fig1.pdf     | `static/images/datasets.png`         |
| fig2.pdf     | `static/images/overview.png` (done)  |
| fig3.pdf     | `static/images/qualitative.png`      |
| fig4.pdf     | `static/images/alpha_schedule.png`   |
| fig5.pdf     | `static/images/architectures.png`    |
| fig6.pdf     | `static/images/cost.png`             |

### PDF → PNG conversion

Pick whichever tool you already have installed. Target DPI 200–300
is a good balance of quality and file size for a web page.

**ImageMagick** (`magick` on Windows / `convert` on Linux-macOS):
```bash
magick -density 300 fig1.pdf -quality 92 docs/static/images/datasets.png
magick -density 300 fig3.pdf -quality 92 docs/static/images/qualitative.png
magick -density 300 fig4.pdf -quality 92 docs/static/images/alpha_schedule.png
magick -density 300 fig5.pdf -quality 92 docs/static/images/architectures.png
magick -density 300 fig6.pdf -quality 92 docs/static/images/cost.png
```

**pdftoppm** (Poppler):
```bash
pdftoppm -r 300 -png fig1.pdf docs/static/images/datasets
# then rename docs/static/images/datasets-1.png -> datasets.png
```

**Python** (`pip install pdf2image` and have Poppler on PATH):
```python
from pdf2image import convert_from_path
for src, dst in [
    ('fig1.pdf', 'datasets.png'),
    ('fig3.pdf', 'qualitative.png'),
    ('fig4.pdf', 'alpha_schedule.png'),
    ('fig5.pdf', 'architectures.png'),
    ('fig6.pdf', 'cost.png'),
]:
    convert_from_path(src, dpi=300)[0].save(f'docs/static/images/{dst}')
```

After conversion, consider compressing the result (TinyPNG, `oxipng`,
or `pngquant`) — the current `overview.png` is ~4 MB, which is fine
but on the heavy side for a project page.

## What to still fill in

Open `index.html` and search for `#` as `href`:

1. **Paper button** — swap `#` for the final PDF URL.
2. **arXiv button** — swap `#` for the arXiv link once posted.
3. **Author profile links** — each `<a href="#">AuthorName</a>` can
   point to a personal homepage or Google Scholar page.
4. **BibTeX entry** — update `year` and `note` once the venue is
   decided; replace with proper `@inproceedings{...}` entry at that
   point.

## Deploying

The site is pure static — any host works.

### GitHub Pages (same repo, `/docs` folder, currently in use)

Repository settings → Pages → Source: `Deploy from a branch` →
Branch: `main`, Folder: `/docs`. The `/docs` folder name is one of
only two options GitHub Pages accepts for "Deploy from a branch" (the
other is repo root).

Live URL: `https://boa-jang.github.io/SGPTTA/`

### Vercel (alternative)

```bash
cd docs
npx vercel          # one-time setup
npx vercel --prod   # subsequent deploys
```

## Local dev tips

- CSS changes are instant on reload.
- Equations are rendered client-side by KaTeX, so you can edit `$...$`
  snippets in `index.html` directly.
- The quantitative tables use a shared `.results-table` class — adjust
  spacing or highlighting in `static/css/style.css`.
