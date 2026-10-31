"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

import PublicAiHub from "@/components/publicModelHub/PublicAiHub";

function PublicAiHubPage() {
  const accessToken = useSearchParams()?.get("key") ?? null;
  return <PublicAiHub accessToken={accessToken} />;
}

export default function PublicModelHub() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center">Loading...</div>}>
      <PublicAiHubPage />
    </Suspense>
  );
}
