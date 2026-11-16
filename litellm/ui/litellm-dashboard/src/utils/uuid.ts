const UUID_V4_FALLBACK = (): string => {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10xx
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};

/**
 * Generate an RFC 4122 version 4 UUID.
 *
 * `crypto.randomUUID` is only exposed in secure contexts (HTTPS or
 * localhost). Gateways are frequently self-hosted on a LAN and served over
 * plain HTTP, where the API is absent, so fall back to a
 * `crypto.getRandomValues`-based implementation, which is available in
 * non-secure contexts as well.
 */
export const uuid = (): string =>
  typeof crypto?.randomUUID === "function" ? crypto.randomUUID() : UUID_V4_FALLBACK();
