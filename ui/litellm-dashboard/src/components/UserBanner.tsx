"use client";

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { Alert } from "antd";

const USER_BANNER_TYPES = ["info", "success", "warning", "error"] as const;
type UserBannerType = (typeof USER_BANNER_TYPES)[number];

const isUserBannerType = (value: unknown): value is UserBannerType => {
  return USER_BANNER_TYPES.includes(value as UserBannerType);
};

export const UserBanner = () => {
  const { data } = useUISettings();
  const values = data?.values ?? {};
  const message =
    typeof values.user_banner_message === "string"
      ? values.user_banner_message.trim()
      : "";

  if (!values.user_banner_enabled || !message) {
    return null;
  }

  const bannerType = isUserBannerType(values.user_banner_type)
    ? values.user_banner_type
    : "info";

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
