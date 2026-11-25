"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { uiHref } from "@/utils/uiHref";

export default function ModelInsightsPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace(`${uiHref("usage")}${window.location.search}${window.location.hash}`);
  }, [router]);

  return <LoadingScreen />;
}
