import { serverRootPath } from "@/lib/serverRootPath";
import { normalizeRootPath } from "@/lib/http/resolveApiBase";

const EXTERNAL_SRC = /^(https?:|data:|blob:|\/\/)/i;

/**
 * Prefix a root-relative asset path (e.g. "/ui/assets/logos/openai.svg") with the
 * proxy's server root path so it resolves when the UI is mounted under a sub-path
 * (e.g. "/litellm" behind a reverse proxy). At the default root it returns the
 * path unchanged.
 */
export const withServerRoot = (path: string, root: string): string => {
  const prefix = normalizeRootPath(root);
  return `${prefix}${path.startsWith("/") ? path : `/${path}`}`;
};

/**
 * Resolve a logo `src` for rendering. External URLs (http(s), data, blob,
 * protocol-relative) are returned untouched; local asset paths are prefixed with
 * the live server root path. `root` is read at call time so it reflects the value
 * `getUiConfig()` resolves asynchronously after load.
 */
export const resolveLogoSrc = (value: string | null | undefined, root: string = serverRootPath): string | undefined => {
  if (!value) return undefined;
  if (EXTERNAL_SRC.test(value)) return value;
  if (value.includes("/_next/")) return value;
  const prefix = normalizeRootPath(root);
  if (prefix && (value === prefix || value.startsWith(`${prefix}/`))) return value;
  return withServerRoot(value, root);
};
