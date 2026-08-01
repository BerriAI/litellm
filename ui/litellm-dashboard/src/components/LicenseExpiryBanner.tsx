"use client";

import React, { useState } from "react";
import { Alert } from "antd";
import { LicenseInfo } from "@/components/networking";
import { useLicenseInfo } from "@/app/(dashboard)/hooks/license/useLicenseInfo";
import { formatExpiryDate, getDaysUntilExpiration, getLicenseExpiryTier } from "@/utils/licenseUtils";

const DISMISS_KEY_PREFIX = "litellm:licenseExpiryBannerDismissed:";
const SALES_EMAIL = "sales@berri.ai";

const salesLink = <a href={`mailto:${SALES_EMAIL}`}>{SALES_EMAIL}</a>;

interface LicenseExpiryBannerProps {
  accessToken: string | null;
}

interface LicenseExpiryBannerViewProps {
  licenseInfo: LicenseInfo | null;
}

const describeCountdown = (days: number): string => {
  if (days <= 0) {
    return "expires today";
  }
  if (days === 1) {
    return "expires in 1 day";
  }
  return `expires in ${days} days`;
};

const expiryDescription = (tier: "warning" | "critical" | "expired"): React.ReactNode => {
  if (tier === "expired") {
    return <>Enterprise features are now disabled. Reach out to {salesLink} to restore access</>;
  }
  if (tier === "critical") {
    return <>Renew now to avoid losing enterprise features. Reach out to {salesLink}</>;
  }
  return <>Renew before it lapses to keep enterprise features. Reach out to {salesLink}</>;
};

export const LicenseExpiryBannerView: React.FC<LicenseExpiryBannerViewProps> = ({ licenseInfo }) => {
  const [locallyDismissed, setLocallyDismissed] = useState(false);

  const expirationDate = licenseInfo?.expiration_date ?? null;
  const tier = getLicenseExpiryTier(expirationDate);
  const days = getDaysUntilExpiration(expirationDate);

  if (expirationDate === null || tier === "none" || days === null) {
    return null;
  }

  const isDismissible = tier === "warning";
  const dismissKey = `${DISMISS_KEY_PREFIX}${expirationDate}`;
  const previouslyDismissed =
    isDismissible && typeof window !== "undefined" ? sessionStorage.getItem(dismissKey) === "true" : false;

  if (isDismissible && (locallyDismissed || previouslyDismissed)) {
    return null;
  }

  const formattedDate = formatExpiryDate(expirationDate);

  const message =
    tier === "expired"
      ? `Your LiteLLM Enterprise license expired on ${formattedDate}`
      : `Your LiteLLM Enterprise license ${describeCountdown(days)} (${formattedDate})`;

  const description = expiryDescription(tier);

  const handleClose = () => {
    if (typeof window !== "undefined") {
      sessionStorage.setItem(dismissKey, "true");
    }
    setLocallyDismissed(true);
  };

  return (
    <Alert
      message={message}
      description={description}
      type={tier === "warning" ? "warning" : "error"}
      showIcon
      banner
      closable={isDismissible}
      onClose={handleClose}
      style={{ marginBottom: 0, borderRadius: 0 }}
    />
  );
};

export const LicenseExpiryBanner: React.FC<LicenseExpiryBannerProps> = ({ accessToken }) => {
  const { data } = useLicenseInfo(accessToken);
  return <LicenseExpiryBannerView licenseInfo={data ?? null} />;
};
