import { proxyBaseUrl } from "@/components/networking";
import { serverRootPath } from "@/lib/serverRootPath";

export function initialProxyAddress(): string {
  const url = new URL(proxyBaseUrl || serverRootPath, window.location.origin);
  if (["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) url.hostname = "host.docker.internal";
  return url.toString().replace(/\/$/, "");
}

export function workerSetupCommand(address: string, token: string, image: string): string {
  const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'";
  return [
    "docker run -d --restart unless-stopped --read-only --cap-drop ALL",
    "  --tmpfs /tmp:rw,noexec,nosuid,size=1g",
    "  --security-opt no-new-privileges --add-host host.docker.internal:host-gateway",
    `  -e ${quote("LITELLM_URL=" + address)}`,
    `  -e ${quote("LENS_WORKER_TOKEN=" + token)}`,
    `  ${quote(image)}`,
  ].join(" \\\n");
}
