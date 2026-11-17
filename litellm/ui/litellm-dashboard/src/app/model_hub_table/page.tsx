"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { useTranslation } from "react-i18next";

import PublicAiHub from "@/components/publicModelHub/PublicAiHub";

function PublicAiHubPage() {
  const accessToken = useSearchParams()?.get("key") ?? null;
  return <PublicAiHub accessToken={accessToken} />;
}

export default function PublicModelHubTable() {
  const { t } = useTranslation();
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center">
          {t("publicModelHub.modelHubTablePage.loading")}
        </div>
      }
    >
      <PublicAiHubPage />
    </Suspense>
  );
}
