// @ts-check

import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';
import { katexMacros } from './src/katex-macros.mjs';
import remarkCallouts from './src/plugins/remark-callouts.mjs';

// https://astro.build/config
export default defineConfig({
	// TODO: replace with your GitHub Pages URL, e.g. https://<username>.github.io
	site: 'https://example.github.io',
	integrations: [mdx(), sitemap()],
	markdown: {
		// remark/rehype pipeline (Astro's default Sätteri processor can't run these plugins).
		processor: unified({
			remarkPlugins: [remarkMath, remarkDirective, remarkCallouts],
			rehypePlugins: [[rehypeKatex, { macros: katexMacros }]],
		}),
	},
});
