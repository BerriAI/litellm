"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelHubTable from "@/components/AIHub/ModelHubTable";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

function PublicModelHubTableContent() {
  const searchParams = useSearchParams()!;
  const key = searchParams.get("key");
  const [accessToken, setAccessToken] = useState<string | null>(null);
  console.log("PublicModelHubTable accessToken:", accessToken);

  useEffect(() => {
    if (!key) {
      return;
    }
    setAccessToken(key);
  }, [key]);

  return (
    <QueryClientProvider client={queryClient}>
      <ModelHubTable accessToken={accessToken} publicPage={true} premiumUser={false} userRole={null} />
    </QueryClientProvider>
  );
}

export default function PublicModelHubTable() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <PublicModelHubTableContent />
    </Suspense>
  );
}
