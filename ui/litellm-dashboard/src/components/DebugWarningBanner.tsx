"use client";

import React from "react";
import { Alert } from "antd";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { Trans, useTranslation } from "react-i18next";

interface DebugWarningBannerProps {
  accessToken: string | null;
}

export const DebugWarningBanner: React.FC<DebugWarningBannerProps> = ({ accessToken }) => {
  const { t } = useTranslation();
  const { data: healthData } = useHealthReadinessDetails(accessToken);

  // Only show banner if detailed debug mode is explicitly enabled
  if (!healthData?.is_detailed_debug) {
    return null;
  }

  return (
    <Alert
      message={t("debugWarningBanner.message")}
      description={<Trans i18nKey="debugWarningBanner.description" components={{ code: <code /> }} />}
      type="warning"
      showIcon
      banner
      style={{ marginBottom: 0, borderRadius: 0 }}
    />
  );
};
