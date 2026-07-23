"use client";

import PromptCachingTab from "@/app/(dashboard)/cost-optimization/_components/PromptCachingTab";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PromptCachingPage() {
  const { accessToken } = useAuthorized();
  return <PromptCachingTab accessToken={accessToken} />;
}
