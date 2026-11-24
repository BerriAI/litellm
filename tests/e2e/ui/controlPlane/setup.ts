import type { FullConfig } from "@playwright/test";

export function credentials() {
  const username = process.env.CONTROL_PLANE_E2E_USERNAME;
  const password = process.env.CONTROL_PLANE_E2E_PASSWORD;
  if (!username || !password) throw new Error("Set CONTROL_PLANE_E2E_USERNAME and CONTROL_PLANE_E2E_PASSWORD.");
  return { username, password };
}

export default async function setup(config: FullConfig) {
  const base = config.projects[0].use.baseURL;
  if (!base) throw new Error("Set a control plane test base URL.");
  const mount = new URL(base);
  const rootPath = decodeURI(mount.pathname).replace(/\/$/, "");
  const discoveryPath = rootPath ? ".well-known/litellm-ui-config" : "litellm/.well-known/litellm-ui-config";
  const discovery = await fetch(new URL(discoveryPath, mount), { redirect: "error" });
  if (!discovery.ok) throw new Error(`Control plane discovery returned ${discovery.status}.`);
  const ui = await discovery.json();
  if (ui.mode !== "control_plane_api") throw new Error("The test must run against a live control plane.");
  if (typeof ui.control_plane_api_prefix !== "string" || !ui.control_plane_api_prefix.startsWith("/"))
    throw new Error("The control plane API prefix must be a same-origin path.");
  if (typeof ui.server_root_path !== "string" || ui.server_root_path.replace(/\/$/, "") !== rootPath)
    throw new Error("The discovery root path does not match the test URL.");
  const prefixPath = `${rootPath}${ui.control_plane_api_prefix}`;
  const prefixUrl = new URL(prefixPath, mount.origin);
  if (prefixUrl.origin !== mount.origin || prefixUrl.pathname !== encodeURI(prefixPath) || prefixUrl.search || prefixUrl.hash)
    throw new Error("The control plane API prefix must be a same-origin path.");
  process.env.CONTROL_PLANE_E2E_API_PREFIX = prefixPath;
  const prefix = prefixUrl.href;
  const login = await fetch(`${prefix}/auth/login`, {
    method: "POST",
    redirect: "error",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials()),
  });
  if (!login.ok) throw new Error(`Control plane test login returned ${login.status}.`);
  const { access_token: token } = await login.json();
  const headers = { Authorization: `Bearer ${token}` };
  const registry = await fetch(`${prefix}/dataplanes`, { headers, redirect: "error" });
  if (!registry.ok) throw new Error(`Dataplane discovery returned ${registry.status}.`);
  const { items } = await registry.json();
  if (items.length !== 2 || items[0].url === items[1].url) throw new Error("Register two independent test dataplanes.");
  const unavailable = process.env.CONTROL_PLANE_E2E_UNREACHABLE;
  if (unavailable && !items.some(({ id }: { id: string }) => id === unavailable)) {
    throw new Error(`The outage test dataplane ${unavailable} is not registered.`);
  }
  for (const item of items) {
    const response = await fetch(`${prefix}/dataplanes/${encodeURIComponent(item.id)}/health`, { headers, redirect: "error" });
    if (!response.ok) throw new Error(`Dataplane ${item.id} health returned ${response.status}.`);
    const { status } = await response.json();
    const expected = item.id === unavailable ? "unreachable" : "ok";
    if (status !== expected) throw new Error(`Dataplane ${item.id} is ${status}, expected ${expected}.`);
  }
}
