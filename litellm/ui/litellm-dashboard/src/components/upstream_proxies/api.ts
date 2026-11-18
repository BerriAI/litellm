import { apiClient } from "@/components/networking";

export const UPSTREAM_PROXY_PROTOCOLS = ["http", "https"] as const;

export type UpstreamProxyProtocol = (typeof UPSTREAM_PROXY_PROTOCOLS)[number];

export interface UpstreamProxy {
  name: string;
  protocol: UpstreamProxyProtocol;
  host: string;
  port: number;
  username: string | null;
  has_password: boolean;
  description: string | null;
  used_by_models: string[];
  used_by_credentials: string[];
}

export interface UpstreamProxyList {
  proxies: UpstreamProxy[];
}

export interface UpstreamProxyAddress {
  protocol: UpstreamProxyProtocol;
  host: string;
  port: number;
  username?: string;
  description?: string;
}

export interface CreateUpstreamProxy extends UpstreamProxyAddress {
  name: string;
  password?: string;
}

export interface UpdateUpstreamProxy extends Partial<UpstreamProxyAddress> {
  password?: string;
}

const UPSTREAM_PROXIES_PATH = "/upstream_proxies";

const upstreamProxyPath = (name: string): string => `${UPSTREAM_PROXIES_PATH}/${encodeURIComponent(name)}`;

export const upstreamProxyListCall = (accessToken: string): Promise<UpstreamProxyList> =>
  apiClient.get<UpstreamProxyList>(UPSTREAM_PROXIES_PATH, { accessToken });

export const upstreamProxyCreateCall = (accessToken: string, proxy: CreateUpstreamProxy): Promise<UpstreamProxyList> =>
  apiClient.post<UpstreamProxyList>(UPSTREAM_PROXIES_PATH, { accessToken, body: proxy });

export const upstreamProxyUpdateCall = (
  accessToken: string,
  name: string,
  update: UpdateUpstreamProxy,
): Promise<UpstreamProxyList> =>
  apiClient.patch<UpstreamProxyList>(upstreamProxyPath(name), { accessToken, body: update });

export const upstreamProxyDeleteCall = (accessToken: string, name: string): Promise<UpstreamProxyList> =>
  apiClient.delete<UpstreamProxyList>(upstreamProxyPath(name), { accessToken });

export const upstreamProxyAddress = (proxy: Pick<UpstreamProxy, "protocol" | "host" | "port">): string =>
  `${proxy.protocol}://${proxy.host.includes(":") ? `[${proxy.host}]` : proxy.host}:${proxy.port}`;
