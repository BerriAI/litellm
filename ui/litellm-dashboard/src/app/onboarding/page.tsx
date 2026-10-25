"use client";
import React, { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { OnboardingForm } from "./OnboardingForm";
import { useLanguage } from "@/contexts/LanguageContext";

function OnboardingContent() {
  const searchParams = useSearchParams()!;
  const action = searchParams.get("action");
  const variant = action === "reset_password" ? "reset_password" : "signup";
  return <OnboardingForm variant={variant} />;
}

export default function Onboarding() {
  const { t } = useLanguage();
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">{t("Loading...")}</div>}>
      <OnboardingContent />
    </Suspense>
  );
}
