// Site-internal URLs, prefixed with the deploy base path (`base` in astro.config.mjs),
// e.g. with base '/blog', url('/posts/foo/') -> '/blog/posts/foo/'. No-op when there's no base.
const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');

export function url(path: string): string {
	return BASE + path;
}

/** Strip the base path, so '/blog/posts/foo/' -> '/posts/foo/'. */
export function stripBase(pathname: string): string {
	return pathname.startsWith(BASE) ? pathname.slice(BASE.length) || '/' : pathname;
}
