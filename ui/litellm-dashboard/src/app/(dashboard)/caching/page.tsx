"use client";

import CacheAnalyticsTab from "./_components/CacheAnalyticsTab";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CachingPage() {
  const { accessToken } = useAuthorized();
  return <CacheAnalyticsTab accessToken={accessToken} />;
}
