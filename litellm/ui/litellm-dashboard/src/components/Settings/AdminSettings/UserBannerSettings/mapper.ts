import type { UserBanner, UserBannerUpdate } from "@/components/networking";

export const toUserBannerFields = ({ enabled, message, severity }: UserBanner): UserBannerUpdate => ({
  enabled,
  message,
  severity,
});
