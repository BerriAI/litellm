"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { modelHubCall } from "@/components/networking";
import ModelHubTable from "@/components/model_hub_table";

export default function PublicModelHubTable() {
  const searchParams = useSearchParams()!;
  const key = searchParams.get("key");
  const [accessToken, setAccessToken] = useState<string | null>(null);

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
  return <ModelHubTable accessToken={accessToken} publicPage={true} premiumUser={false} userRole={null} />;
}
