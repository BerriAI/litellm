import React from "react";
import { Alert, Button } from "antd";
import { getLoginUrl } from "@/utils/returnUrlUtils";
import { useTranslation } from "react-i18next";

export function OnboardingErrorView() {
  const { t } = useTranslation("auth");
  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Alert
        type="error"
        message={t("onboarding.loadErrorTitle")}
        description={t("onboarding.loadErrorDescription")}
        showIcon
      />
      <div className="mt-4">
        <Button href={getLoginUrl()}>{t("onboarding.backToLogin")}</Button>
      </div>
    </div>
  );
}
