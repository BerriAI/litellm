"use client";

import PromptCompressionTab from "@/app/(dashboard)/cost-optimization/_components/PromptCompressionTab";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PromptCompressionPage() {
  const { accessToken } = useAuthorized();
  return <PromptCompressionTab accessToken={accessToken} />;
}
