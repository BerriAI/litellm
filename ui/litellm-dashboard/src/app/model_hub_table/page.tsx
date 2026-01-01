"use client";
import React, { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelHubTable from "@/components/AIHub/ModelHubTable";

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
  return <ModelHubTable accessToken={accessToken} publicPage={true} premiumUser={false} userRole={null} />;
}
