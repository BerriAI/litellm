"use client";
import React, { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { OnboardingForm } from "./OnboardingForm";
import LanguageSelector from "@/components/LanguageSelector/LanguageSelector";
import { useTranslation } from "react-i18next";

function OnboardingContent() {
  const searchParams = useSearchParams()!;
  const action = searchParams.get("action");
  const variant = action === "reset_password" ? "reset_password" : "signup";
  return <OnboardingForm variant={variant} />;
}

export default function Onboarding() {
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
      <OnboardingContent />
    </Suspense>
  );
}
