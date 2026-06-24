export interface ApiBaseInputs {
  /**
   * Absolute API origin to target instead of same-origin. Comes from the
   * NEXT_PUBLIC_BASE_URL build env (dev / split-origin) or the proxy_base_url
   * the backend reports in its UI config. Empty/unset means same-origin.
   */
  explicitBase?: string | null;
  /**
   * The proxy's mount path when it sits under a sub-path (e.g. "/litellm"
   * behind a reverse proxy). "/" or empty means mounted at the root.
   */
  serverRootPath?: string | null;
}

const normalizeRootPath = (serverRootPath: string | null | undefined): string => {
  const trimmed = (serverRootPath ?? "").trim();
  if (trimmed === "" || trimmed === "/") return "";
  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withLeadingSlash.replace(/\/+$/, "");
};

/**
 * Resolve the single API base string that every request is prefixed with.
 *
 * Same-origin is represented as "" so requests stay relative and work behind
 * any domain or reverse proxy without hardcoding an origin. An explicit base is
 * used verbatim (trailing slash trimmed). The server root path is appended
 * unless the base already ends with it.
 */
export const resolveApiBase = ({ explicitBase, serverRootPath }: ApiBaseInputs): string => {
  const base = (explicitBase ?? "").trim().replace(/\/+$/, "");
  const rootPath = normalizeRootPath(serverRootPath);
  if (rootPath === "" || base.endsWith(rootPath)) return base;
  return `${base}${rootPath}`;
};
