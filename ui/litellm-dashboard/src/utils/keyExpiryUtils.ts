const HAS_TIMEZONE_SUFFIX = /[zZ]$|[+-]\d{2}:?\d{2}$/;

/**
 * Parse a key expires timestamp as UTC, matching proxy auth behavior for naive DB values.
 */
export function parseExpiresUtc(expires: string): number {
  const normalized = HAS_TIMEZONE_SUFFIX.test(expires) ? expires : `${expires}Z`;
  return Date.parse(normalized);
}

/**
 * Returns true when the key has an expires value in the past (UTC).
 */
export function isKeyExpired(expires?: string | null): boolean {
  if (!expires) {
    return false;
  }

  const expiryMs = parseExpiresUtc(expires);
  return !Number.isNaN(expiryMs) && expiryMs < Date.now();
}
