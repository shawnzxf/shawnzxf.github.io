import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const posts = defineCollection({
	loader: glob({ base: './src/content/posts', pattern: '**/*.{md,mdx}' }),
	schema: z.object({
		title: z.string(),
		// One-line summary: home page card, post subtitle, and SEO/link-preview description.
		summary: z.string(),
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		tags: z.array(z.string()).default([]),
		// KaTeX shortcuts for this post only, e.g. { '\\R': '\\mathbb{R}' }.
		macros: z.record(z.string(), z.string()).optional(),
		// Drafts render in `npm run dev` but are excluded from production builds.
		draft: z.boolean().default(false),
	}),
});

export const collections = { posts };
