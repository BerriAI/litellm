"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ModelInsightsView from "./_components/ModelInsightsView";

export default function ModelInsightsPage() {
  const { accessToken } = useAuthorized();
  return <ModelInsightsView accessToken={accessToken} />;
}
