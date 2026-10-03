import { proxyBaseUrl } from "@/components/networking";
import { serverRootPath } from "@/lib/serverRootPath";

export const LENS_WORKER_IMAGE =
  "ghcr.io/berriai/litellm-lens-worker@sha256:44f0597c7583dcfef999ece9a8bc02cfeb9f0f5167a1221cee3bd10b1b79271b";

export function initialProxyAddress(): string {
  const url = new URL(proxyBaseUrl || serverRootPath, window.location.origin);
  if (["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) url.hostname = "host.docker.internal";
  return url.toString().replace(/\/$/, "");
}

export function workerSetupCommand(address: string, token: string): string {
  const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'";
  return [
    "docker run -d --restart unless-stopped --read-only --cap-drop ALL",
    "  --tmpfs /tmp:rw,noexec,nosuid,size=1g",
    "  --security-opt no-new-privileges --platform linux/amd64 --add-host host.docker.internal:host-gateway",
    `  -e ${quote("LITELLM_URL=" + address)}`,
    `  -e ${quote("LENS_WORKER_TOKEN=" + token)}`,
    `  ${LENS_WORKER_IMAGE}`,
  ].join(" \\\n");
}
