// Per-post KaTeX macros, declared in a post's frontmatter:
//
//   macros:
//     '\R': '\mathbb{R}'
//     '\T': '^{\top}'
//
// Returns a fresh object per call so `\gdef` in one post can't leak into another.
export function postMacros(file) {
	const macros = file.data.astro?.frontmatter?.macros;
	return macros && typeof macros === 'object' ? { ...macros } : {};
}
