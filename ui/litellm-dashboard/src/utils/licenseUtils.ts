export type LicenseExpiryTier = "none" | "warning" | "critical" | "expired";

export const LICENSE_EXPIRY_WARNING_DAYS = 30;
export const LICENSE_EXPIRY_CRITICAL_DAYS = 7;

export const getDaysUntilExpiration = (expirationDate: string | null, now: Date = new Date()): number | null => {
  if (!expirationDate) {
    return null;
  }

  const expiration = new Date(`${expirationDate}T00:00:00Z`);
  if (Number.isNaN(expiration.getTime())) {
    return null;
  }

  const nowUtcMidnight = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const diffMs = expiration.getTime() - nowUtcMidnight;
  return Math.ceil(diffMs / (1000 * 60 * 60 * 24));
};

export const getLicenseExpiryTier = (expirationDate: string | null, now: Date = new Date()): LicenseExpiryTier => {
  const days = getDaysUntilExpiration(expirationDate, now);
  if (days === null) {
    return "none";
  }
  if (days < 0) {
    return "expired";
  }
  if (days <= LICENSE_EXPIRY_CRITICAL_DAYS) {
    return "critical";
  }
  if (days <= LICENSE_EXPIRY_WARNING_DAYS) {
    return "warning";
  }
  return "none";
};

const EXPIRY_DATE_FORMAT: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
};

export const formatExpiryDate = (expirationDate: string): string => {
  const date = new Date(`${expirationDate}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) {
    return expirationDate;
  }
  return date.toLocaleDateString("en-US", EXPIRY_DATE_FORMAT);
};
