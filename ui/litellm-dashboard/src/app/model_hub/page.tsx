"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { modelHubCall } from "@/components/networking";
import ModelHub from "@/components/model_hub";

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
  return (
    <ModelHub accessToken={accessToken} publicPage={true} premiumUser={false} />
  );
}
