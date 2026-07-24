"use client";

import CacheSettings from "@/app/(dashboard)/caching/_components/cache_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CacheSettingsPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <CacheSettings accessToken={accessToken} userRole={userRole} userID={userId} />;
}
