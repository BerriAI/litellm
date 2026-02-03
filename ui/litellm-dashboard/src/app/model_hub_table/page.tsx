"use client";
import React, { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelHubTable from "@/components/AIHub/ModelHubTable";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

export default function PublicModelHubTable() {
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
  /**
   * populate navbar
   *
   */
  return (
    <QueryClientProvider client={queryClient}>
      <ModelHubTable accessToken={accessToken} publicPage={true} premiumUser={false} userRole={null} />
    </QueryClientProvider>
  );
}
