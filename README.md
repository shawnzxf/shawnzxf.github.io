# Personal blog

Astro static site with KaTeX math, deployed to GitHub Pages.

## Develop

```sh
npm install
npm run dev       # http://localhost:4321, drafts visible
npm run build     # output in dist/, drafts excluded
npm run preview   # serve dist/
```

## Write a post

Add `src/content/posts/<slug>.md`; it is served at `/posts/<slug>/`.

```markdown
---
title: 'Post title'
summary: 'One-line summary: shown on the home page card, under the post title, and in search/link previews.'
pubDate: 2026-10-24
tags: [ml, math]
draft: true        # optional; hidden in production
---

Inline math $E = mc^2$, display math:

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}
$$
```

Prefer `.md` over `.mdx` for math-heavy posts (`{` and `<` are special in MDX). Write a literal dollar sign as `\$`.

### Callout boxes

```markdown
:::blue[Optional label]
Blue left border. Also `:::green`, `:::amber`, and `:::neutral` (plain gray box).
:::

:::muted
Muted side note (small gray text, no box). Inside a table cell use `<span class="muted">…</span>`.
:::
```

To nest a callout inside another, give the outer one more colons (`::::amber` … `::::`).

### KaTeX macros

Shortcuts are per post: declare them in the post's frontmatter and they apply to that post only.

```yaml
macros:
  '\R': '\mathbb{R}'
  '\T': '^{\top}'
```

After changing anything under `src/plugins/` or `astro.config.mjs`, restart the dev server with `npx astro dev --force` (Astro caches rendered posts).

## Deploy

1. Create a GitHub repo named `<username>.github.io` and push this project to `main`.
2. In the repo: **Settings → Pages → Source: GitHub Actions**.
3. Set `site` in `astro.config.mjs` and personalize `src/consts.ts`.

Every push to `main` runs `.github/workflows/deploy.yml` and publishes the site.
