"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelHubTable from "@/components/AIHub/ModelHubTable";
import LanguageSelector from "@/components/LanguageSelector/LanguageSelector";
import { useTranslation } from "react-i18next";

function PublicModelHubTableContent() {
  const searchParams = useSearchParams()!;
  const key = searchParams.get("key");
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    if (!key) {
      return;
    }
    setAccessToken(key);
  }, [key]);

  return <ModelHubTable accessToken={accessToken} publicPage={true} premiumUser={false} userRole={null} />;
}

export default function PublicModelHubTable() {
  const { t } = useTranslation("common");
  return (
    <>
      <div className="fixed right-4 top-4 z-50 rounded-lg border border-gray-200 bg-white p-1 shadow-sm">
        <LanguageSelector />
      </div>
      <Suspense fallback={<div className="flex items-center justify-center min-h-screen">{t("states.loading")}</div>}>
        <PublicModelHubTableContent />
      </Suspense>
    </>
  );
}
