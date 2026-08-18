"use client";

import ApiKeysDashboard from "@/app/(dashboard)/api-keys/ApiKeysDashboard";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { Suspense } from "react";

function ApiKeysPageContent() {
  const { isLoading, isAuthorized } = useAuthorized();
  if (isLoading || !isAuthorized) {
    return <LoadingScreen />;
  }
  return <ApiKeysDashboard />;
}

export default function ApiKeysPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <ApiKeysPageContent />
    </Suspense>
  );
}
