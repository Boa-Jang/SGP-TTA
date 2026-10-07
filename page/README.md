# SGP-TTA Project Page

Static, single-file project page for SGP-TTA. No build step; everything
loads from CDNs. Open `index.html` locally to preview.

```bash
# from the repository root
cd page
python -m http.server 8000
# then visit http://localhost:8000
```

## Layout

```
page/
├── index.html            # the whole page
├── static/
│   ├── css/style.css     # minor overrides on top of Bulma
│   └── images/
│       ├── overview.png                 # method overview figure
│       ├── results/                     # qualitative-comparison overlays
│       │   ├── drive_source.png
│       │   ├── drive_sgptta.png
│       │   ├── octa_source.png
│       │   ├── octa_sgptta.png
│       │   ├── stare_source.png
│       │   └── stare_sgptta.png
│       └── comparisons/                 # extra side-by-side crops (optional)
└── README.md             # this file
```

## What to fill in before publishing

Open `index.html` and search for `TODO:` — six placeholders:

1. **Authors** (hero block) — names, affiliations, superscripts.
2. **Venue** (`Preprint, 2026`) — swap to the final venue once accepted.
3. **Paper / arXiv / Code** button URLs — currently `#`.
4. **Abstract** — paste the camera-ready abstract.
5. **Method figures** — right now both method sub-sections reuse
   `overview.png`. Replace with a ProgBN schedule figure and a CSR
   figure once available.
6. **Quantitative Results table** — the Dice / clDice numbers are
   currently `—`.

## Preparing qualitative-result overlays

The `Qualitative Results` section uses
[img-comparison-slider](https://img-comparison-slider.com/) and expects
two aligned overlay images per target domain under
`static/images/results/`:

| Target | Left slot (Source) | Right slot (SGP-TTA) |
| ------ | ------------------ | -------------------- |
| DRIVE  | `drive_source.png` | `drive_sgptta.png`   |
| OCTA   | `octa_source.png`  | `octa_sgptta.png`    |
| STARE  | `stare_source.png` | `stare_sgptta.png`   |

Each image should be the same crop of the RGB input with the predicted
vessel mask overlaid in a single color (e.g. the mask drawn in red on
top of the input), so the slider seams line up. The current
`script/inference.py` saves a combined 5-panel figure; to produce
slider-ready overlays specifically, extend the script or save the
individual overlays from the notebook.

## Deploying

The site is pure static — any host works.

### Vercel

```bash
# one-time (choose "Deploy from current directory"):
cd page
npx vercel

# subsequent deploys:
npx vercel --prod
```

### GitHub Pages (same repo, `/page` folder)

Repository settings → Pages → Source: `main` branch, `/page` folder.

### GitHub Pages (separate `username.github.io/sgptta` repo)

Copy the contents of `page/` into a new repo named `sgptta` and enable
Pages on that repo (`main` branch, root folder).

## Local dev tips

- CSS changes are instant on reload.
- Equations are rendered client-side by KaTeX, so you can edit `$...$`
  snippets in `index.html` directly.
- The comparison slider is a web component — no extra JS plumbing.
