"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { Suspense } from "react";
import SecureShareView from "@/components/secure_share/SecureShareView";

function SecureShareViewPageContent() {
  const { isLoading, isAuthorized, accessToken } = useAuthorized();
  if (isLoading || !isAuthorized) {
    return <LoadingScreen />;
  }
  return <SecureShareView accessToken={accessToken} />;
}

export default function SecureShareViewPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <SecureShareViewPageContent />
    </Suspense>
  );
}
