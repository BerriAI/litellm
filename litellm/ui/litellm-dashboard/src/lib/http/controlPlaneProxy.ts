import { controlPlaneResourceSegment } from "./controlPlane";

type ProxySession = {
  prefix: string;
  dataplane: string | null;
  token: string;
  canRead: boolean;
  canWrite: boolean;
  expire: () => void;
};

let session: ProxySession | null = null;

export function registerControlPlaneProxy(value: ProxySession) {
  session = value;
  return () => {
    if (session === value) session = null;
  };
}

export const controlPlaneProxyOrigin = () => (session ? window.location.origin : null);

export function captureProxySession() {
  const startingSession = session;
  return () => session === startingSession;
}

function rejected(status: number, detail: string): Response {
  return Response.json({ detail }, { status });
}

export const proxyFetch: typeof fetch = async (...args) => {
  const active = session;
  if (!active) return globalThis.fetch(...args);
  const [input, init] = args;
  const request = new Request(input instanceof Request ? input : new URL(input, window.location.origin), init);
  const path = new URL(request.url);
  if (path.origin !== window.location.origin) {
    return rejected(403, "Cross-origin requests aren't available through the control plane.");
  }
  const read = request.method === "GET" || request.method === "HEAD";
  if (!active.dataplane || !(read ? active.canRead : active.canWrite)) {
    return rejected(403, "This session cannot use that dataplane management action.");
  }
  const headers = new Headers();
  for (const name of ["accept", "content-type", "if-match", "if-none-match", "idempotency-key"]) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  headers.set("Authorization", `Bearer ${active.token}`);
  const url = `${active.prefix}/dataplanes/${controlPlaneResourceSegment(active.dataplane)}/proxy${path.pathname}${path.search}`;
  const options: RequestInit = {
    method: request.method,
    headers,
    body: request.body === null ? undefined : await request.arrayBuffer(),
    signal: request.signal,
    cache: "no-store",
    credentials: "omit",
    redirect: "error",
  };
  const response = await globalThis.fetch(url, options);
  if (session === active && response.status === 401 && !response.headers.has("X-LiteLLM-Dataplane")) active.expire();
  return response;
};
