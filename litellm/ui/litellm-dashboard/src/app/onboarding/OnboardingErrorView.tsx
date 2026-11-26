import React from "react";
import { useTranslation } from "react-i18next";
import { CircleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { buttonVariants } from "@/components/ui/button";
import { getProxyBaseUrl } from "@/components/networking";
import { getLoginUrl } from "@/utils/returnUrlUtils";

export function OnboardingErrorView() {
  const { t } = useTranslation();
  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Alert variant="error">
        <CircleAlert />
        <AlertTitle>{t("onboarding.errorView.title")}</AlertTitle>
        <AlertDescription>{t("onboarding.errorView.description")}</AlertDescription>
      </Alert>
      <div className="mt-4">
        <a href={getLoginUrl(getProxyBaseUrl())} className={buttonVariants({ variant: "outline" })}>
          {t("onboarding.errorView.back")}
        </a>
      </div>
    </div>
  );
}
