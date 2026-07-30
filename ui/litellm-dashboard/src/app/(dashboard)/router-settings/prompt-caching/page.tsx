"use client";

import { PromptCachingSettingsTab } from "@/app/(dashboard)/router-settings/_components/general_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PromptCachingPage() {
  const { accessToken } = useAuthorized();
  if (!accessToken) {
    return null;
  }
  return <PromptCachingSettingsTab accessToken={accessToken} />;
}
