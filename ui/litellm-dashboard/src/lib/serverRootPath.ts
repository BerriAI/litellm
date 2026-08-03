/**
 * Canonical holder for the proxy's server root path (e.g. "/litellm" when mounted
 * behind a reverse proxy, "/" at the root). Kept in a tiny leaf module so url/asset
 * helpers can read it without importing — and forcing every test to mock — the large
 * networking module. `networking` re-exports `serverRootPath` and updates it via
 * `setServerRootPath` from its UI-config bootstrap.
 */
export let serverRootPath = "/";

export const setServerRootPath = (rootPath: string): void => {
  serverRootPath = rootPath;
};
