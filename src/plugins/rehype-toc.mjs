// Collects h2–h4 headings into `frontmatter.toc` for the post sidebar.
//
// Runs before rehype-katex, so inline math is still `<code class="math-inline">TeX</code>`.
// That lets us (1) build clean heading ids from the TeX source instead of KaTeX's
// triplicated output text, and (2) render the math in the TOC labels ourselves.
import Slugger from 'github-slugger';
import katex from 'katex';
import { visit } from 'unist-util-visit';
import { postMacros } from './post-macros.mjs';

const escapeHtml = (s) =>
	s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const isInlineMath = (node) =>
	node.type === 'element' &&
	node.tagName === 'code' &&
	[].concat(node.properties?.className ?? []).includes('math-inline');

export default function rehypeToc() {
	return (tree, file) => {
		const macros = postMacros(file);
		const slugger = new Slugger();
		const toc = [];

		visit(tree, 'element', (node) => {
			const depth = { h2: 2, h3: 3, h4: 4 }[node.tagName];
			if (!depth) return;

			let text = '';
			let html = '';
			const walk = (n) => {
				if (n.type === 'text') {
					text += n.value;
					html += escapeHtml(n.value);
				} else if (isInlineMath(n)) {
					const tex = n.children.map((c) => c.value ?? '').join('');
					text += tex;
					html += katex.renderToString(tex, { macros, throwOnError: false, output: 'html' });
				} else if (n.children) {
					n.children.forEach(walk);
				}
			};
			node.children.forEach(walk);

			node.properties ||= {};
			const slug = typeof node.properties.id === 'string' ? node.properties.id : slugger.slug(text);
			node.properties.id = slug;
			toc.push({ depth, slug, html });
		});

		const astro = (file.data.astro ||= {});
		(astro.frontmatter ||= {}).toc = toc;
	};
}
