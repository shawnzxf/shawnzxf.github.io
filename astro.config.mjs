// @ts-check

import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';
import rehypeKatexPost from './src/plugins/rehype-katex-post.mjs';
import remarkCallouts from './src/plugins/remark-callouts.mjs';
import rehypeToc from './src/plugins/rehype-toc.mjs';

// https://astro.build/config
export default defineConfig({
	// TODO: replace with your GitHub Pages URL, e.g. https://<username>.github.io
	site: 'https://xiufengzhao.github.io',
	integrations: [mdx(), sitemap()],
	markdown: {
		// remark/rehype pipeline (Astro's default Sätteri processor can't run these plugins).
		processor: unified({
			remarkPlugins: [remarkMath, remarkDirective, remarkCallouts],
			// rehypeToc must run before KaTeX (it reads the TeX source of inline math).
			// KaTeX macros are per post, from frontmatter `macros`.
			rehypePlugins: [rehypeToc, rehypeKatexPost],
		}),
	},
});
