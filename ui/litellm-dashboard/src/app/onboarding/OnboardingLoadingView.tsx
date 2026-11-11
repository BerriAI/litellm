import React from "react";
import { useTranslation } from "react-i18next";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

export function OnboardingLoadingView() {
  const { t } = useTranslation();
  return (
    <div className="mx-auto w-full max-w-md mt-10 flex justify-center">
      <UiLoadingSpinner
        role="status"
        aria-label={t("onboarding.loadingView.label")}
        className="size-8 text-muted-foreground"
      />
    </div>
  );
}
