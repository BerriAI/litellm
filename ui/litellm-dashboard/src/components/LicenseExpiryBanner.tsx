"use client";

import React, { useEffect, useState } from "react";
import { Alert } from "antd";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getLicenseInfo, LicenseInfo } from "./networking";

const EXPIRY_WARNING_DAYS = 14;
/** Re-fetch license info every 6 hours so long-lived sessions pick up expiry. */
const REFRESH_INTERVAL_MS = 6 * 60 * 60 * 1000;

export const LicenseExpiryBanner: React.FC = () => {
  const { accessToken } = useAuthorized();
  const [licenseInfo, setLicenseInfo] = useState<LicenseInfo | null>(null);
  const [fetchError, setFetchError] = useState(false);

  useEffect(() => {
    if (!accessToken) return;

    let cancelled = false;
    let inFlight = false;

    const fetchLicense = () => {
      if (inFlight) return; // skip if a previous request is still pending
      inFlight = true;
      getLicenseInfo(accessToken)
        .then((info) => {
          if (!cancelled) {
            setLicenseInfo(info);
            setFetchError(false);
          }
        })
        .catch(() => {
          if (!cancelled) setFetchError(true);
        })
        .finally(() => {
          inFlight = false;
        });
    };

    fetchLicense();
    const intervalId = setInterval(fetchLicense, REFRESH_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [accessToken]);

  // Only show a fetch-error banner if we previously confirmed this is an
  // enterprise license holder. This prevents non-enterprise users from seeing
  // a misleading "Unable to verify enterprise license" warning on network errors.
  if (fetchError && licenseInfo?.has_license) {
    return (
      <Alert
        message="Unable to verify enterprise license"
        description="Could not reach the license server. If your license is near expiry, you may not see a warning. Please check your connection or contact support."
        type="warning"
        showIcon
        banner
        closable={false}
        style={{ marginBottom: 0, borderRadius: 0 }}
      />
    );
  }

  if (!licenseInfo?.has_license || !licenseInfo.expiration_date) {
    return null;
  }

  // Strip any existing time component so we always get a valid date-only base,
  // then append end-of-day in UTC to avoid timezone-dependent parsing.
  const dateOnly = licenseInfo.expiration_date.split("T")[0];
  const expDate = new Date(dateOnly + "T23:59:59Z");

  // Guard against unparseable expiration_date values (e.g. "N/A", malformed strings)
  if (isNaN(expDate.getTime())) {
    return null;
  }

  const now = new Date();
  const diffMs = expDate.getTime() - now.getTime();
  const daysRemaining = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

  if (daysRemaining > EXPIRY_WARNING_DAYS) {
    return null;
  }

  const expirationDisplay = expDate.toLocaleString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });

  const isExpired = daysRemaining <= 0;
  const message = isExpired
    ? "Enterprise License Expired"
    : `Enterprise License Expiring in ${daysRemaining} day${daysRemaining === 1 ? "" : "s"}`;

  const description = isExpired
    ? `Your enterprise license expired on ${expirationDisplay}. Please contact support to renew.`
    : `Your enterprise license will expire on ${expirationDisplay}. Please contact support to renew before expiration.`;

  return (
    <Alert
      message={message}
      description={description}
      type={isExpired ? "error" : "warning"}
      showIcon
      banner
      closable={false}
      style={{ marginBottom: 0, borderRadius: 0 }}
    />
  );
};
