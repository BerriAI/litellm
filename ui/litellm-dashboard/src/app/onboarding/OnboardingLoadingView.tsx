import React from "react";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useLanguage } from "@/contexts/LanguageContext";

export function OnboardingLoadingView() {
  const { t } = useLanguage();
  return (
    <div className="mx-auto w-full max-w-md mt-10 flex justify-center">
      <UiLoadingSpinner role="status" aria-label={t("Loading invitation")} className="size-8 text-muted-foreground" />
    </div>
  );
}
