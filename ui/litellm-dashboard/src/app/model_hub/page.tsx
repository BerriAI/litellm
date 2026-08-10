"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import PublicModelHubPage from "@/components/public_model_hub";
import LanguageSelector from "@/components/LanguageSelector/LanguageSelector";
import { useTranslation } from "react-i18next";

function PublicModelHubContent() {
  const searchParams = useSearchParams()!;
  const key = searchParams.get("key");
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    if (!key) {
      return;
    }
    setAccessToken(key);
  }, [key]);

  return <PublicModelHubPage accessToken={accessToken} />;
}

export default function PublicModelHub() {
  const { t } = useTranslation("common");
  return (
    <Suspense
      fallback={
        <>
          <div className="fixed right-4 top-4 z-50 rounded-lg border border-gray-200 bg-white p-1 shadow-sm">
            <LanguageSelector />
          </div>
          <div className="flex items-center justify-center min-h-screen">{t("states.loading")}</div>
        </>
      }
    >
      <PublicModelHubContent />
    </Suspense>
  );
}
