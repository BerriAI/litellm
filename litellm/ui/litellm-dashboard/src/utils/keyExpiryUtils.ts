import { add } from "date-fns";

const HAS_TIMEZONE_SUFFIX = /[zZ]$|[+-]\d{2}:?\d{2}$/;

/**
 * Parse a key expires timestamp as UTC, matching proxy auth behavior for naive DB values.
 */
export function parseExpiresUtc(expires: string): number {
  const normalized = HAS_TIMEZONE_SUFFIX.test(expires) ? expires : `${expires}Z`;
  return Date.parse(normalized);
}

/**
 * Format an expires timestamp for display, treating naive DB values as UTC.
 */
export function formatExpiresUtc(expires: string): string {
  const expiryMs = parseExpiresUtc(expires);
  if (Number.isNaN(expiryMs)) {
    return expires;
  }

  return new Date(expiryMs).toLocaleString();
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

/**
 * Compute a human-readable preview of the new expiry from a duration string.
 * Returns null when the duration cannot be parsed.
 */
export function calculateExpiryPreviewFromDuration(duration: string | undefined): string | null {
  if (!duration) {
    return null;
  }

  try {
    const amount = parseInt(duration);
    if (Number.isNaN(amount)) {
      throw new Error("Invalid duration format");
    }

    const now = new Date();
    // Check "mo" before "m" to avoid a false prefix match (e.g. "1mo" → minutes).
    let newExpiry: Date;
    if (duration.endsWith("mo")) {
      newExpiry = add(now, { months: amount });
    } else if (duration.endsWith("s")) {
      newExpiry = add(now, { seconds: amount });
    } else if (duration.endsWith("m")) {
      newExpiry = add(now, { minutes: amount });
    } else if (duration.endsWith("h")) {
      newExpiry = add(now, { hours: amount });
    } else if (duration.endsWith("d")) {
      newExpiry = add(now, { days: amount });
    } else if (duration.endsWith("w")) {
      newExpiry = add(now, { weeks: amount });
    } else {
      throw new Error("Invalid duration format");
    }

    return newExpiry.toLocaleString();
  } catch {
    return null;
  }
}
