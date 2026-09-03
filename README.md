# Yang Qianxi · Portfolio

Personal research portfolio for 杨骞玺 (Yang Qianxi) — robotics engineering undergraduate at Beijing Institute of Technology. Built for research internship / graduate-school applications in robotics & embodied AI.

Plain **HTML + CSS + JS**, zero dependencies, no build step. Deployable to any static host (GitHub Pages, etc.).

## Files

| File | What it is |
|---|---|
| `index.html` | The whole site (single page, all sections) |
| `styles.css` | Design tokens + all styling |
| `main.js` | Language toggle, mobile menu, scroll-spy, reveal animations |
| `assets/avatar.svg` | Initials avatar — swap for a photo if you want |
| `assets/cv.pdf` | Copy of your resume (the "CV ↓" buttons point here) |
| `design_reference_img/` | The original design mockups this site was built from (do not delete) |
| `resume_file/` | Your original resume PDF (do not delete) |

## Run locally

```bash
cd Qx_Blog
python3 -m http.server 8000
# open http://localhost:8000
```

Or just double-click `index.html` — it works from `file://` too (Google Fonts need internet; everything falls back to system fonts offline).

## Editing text (bilingual)

Every translatable element appears **twice**, once per language:

```html
<p data-i18n="en">English text here</p>
<p data-i18n="zh">中文文本</p>
```

- English shows by default; the nav toggle switches to Chinese.
- `main.js` remembers the choice in `localStorage`.
- Numbers, code, and labels that don't need translation appear only once.
- Untranslatable element? Just give it **no** `data-i18n` attribute.

## Adding a project demo video

Find a card's media slot:

```html
<div class="media-slot" data-video-slot>
  <svg class="play-icon">…</svg>
  <span class="media-text">…</span>
</div>
```

Replace the inner content with:

```html
<video controls preload="none" poster="assets/demo_act.jpg" style="width:100%;height:100%;object-fit:cover">
  <source src="assets/videos/act_peg_in_hole.mp4" type="video/mp4">
</video>
```

Drop the file into `assets/videos/` (or use any video URL / Bilibili iframe). Done — the card keeps its layout.

## Adding a research note

Copy a `.note-card` block inside the right `.note-group`, then change the title and link the card to the note (Xiaohongshu URL or a local page). The section is organized by category: robot learning / embodied AI / research practice / math.

## Adding an internship entry

In `#experience`, copy the `.tl-item.tl-ghost` block and fill it in. Suggested fields: organization & role · dates · the problem you owned · your contribution · stack · quantitative result.

## Swapping the avatar photo

Replace `assets/avatar.svg` with `assets/avatar.jpg` (square, ≥400×400) and update the `<img src>` in the hero. That's it.

## URLs to fill in (search for `TODO` in `index.html`)

1. **Xiaohongshu** — 2 places: the "Follow my learning notes" button and the footer button (currently link to `xiaohongshu.com` homepage).
2. **Hugging Face** — the ACT project's weights link.
3. Optional: add GitHub / Hugging Face footer buttons (commented template is already in the footer).

## Deploy to GitHub Pages

```bash
cd Qx_Blog
git init
git add .
git commit -m "Initial portfolio"
# create a repo on GitHub (e.g. yourname.github.io), then:
git remote add origin https://github.com/YOU/YOU.github.io.git
git push -u origin main
# Settings → Pages → deploy from branch: main / root
```

## Design tokens

Measured from `design_reference_img/` — keep these when adding styles:

```css
--bg: #F7F8FA;   --card: #FEFEFE;   --surface: #F1F2F6;
--ink: #18191B;  --body: #393B42;   --muted: #AAACB5;
--accent: #6795ED;  --accent-ink: #3B5B9E;
--lime: #BBD955;  --amber: #EDA304;  --teal: #80D8DA;  /* per-project accents */
```
