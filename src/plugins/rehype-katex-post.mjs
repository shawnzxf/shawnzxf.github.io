// rehype-katex, but using each post's own frontmatter `macros` (see post-macros.mjs).
import rehypeKatex from 'rehype-katex';
import { postMacros } from './post-macros.mjs';

export default function rehypeKatexPost(options = {}) {
	return (tree, file) => rehypeKatex({ ...options, macros: postMacros(file) })(tree, file);
}
