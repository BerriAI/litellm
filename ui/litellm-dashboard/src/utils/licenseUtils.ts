import i18n from "@/i18n/i18n";

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

export const formatExpirationStatus = (expirationDate: string | null, now: Date = new Date()): string => {
  const days = getDaysUntilExpiration(expirationDate, now);
  if (expirationDate === null || days === null) {
    return i18n.t("utils.licenseUtils.noExpiration");
  }
  return days < 0
    ? i18n.t("utils.licenseUtils.expired", { date: formatExpiryDate(expirationDate) })
    : i18n.t("utils.licenseUtils.expires", { date: formatExpiryDate(expirationDate) });
};
