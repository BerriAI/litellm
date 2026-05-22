"use client";

import React from "react";
import { Alert } from "antd";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";

const ALLOWED_BANNER_TYPES = ["info", "success", "warning", "error"] as const;
type BannerType = (typeof ALLOWED_BANNER_TYPES)[number];

function isBannerType(value: unknown): value is BannerType {
  return ALLOWED_BANNER_TYPES.includes(value as BannerType);
}

export const UserBanner: React.FC = () => {
  const { data } = useUISettings();
  const values = data?.values ?? {};

  const enabled = Boolean(values.user_banner_enabled);
  const message =
    typeof values.user_banner_message === "string"
      ? values.user_banner_message.trim()
      : "";
  const rawType = values.user_banner_type;
  const bannerType: BannerType = isBannerType(rawType) ? rawType : "info";

  if (!enabled || !message) {
    return null;
  }

  return (
    <Alert
      message={message}
      type={bannerType}
      showIcon
      banner
      style={{ marginBottom: 0, borderRadius: 0 }}
    />
  );
};

export default UserBanner;
