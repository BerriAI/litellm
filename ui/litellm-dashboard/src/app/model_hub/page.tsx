"use client";
import React, { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import PublicModelHubPage from "@/components/public_model_hub";

export default function PublicModelHub() {
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
  return <PublicModelHubPage accessToken={accessToken} />;
}
