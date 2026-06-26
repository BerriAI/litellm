import React from "react";
import { Alert, Button } from "antd";
import { useTranslation } from "react-i18next";

export function OnboardingErrorView() {
  const { t } = useTranslation();
  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Alert
        type="error"
        message={t("pages.onboardingErrorView.message")}
        description={t("pages.onboardingErrorView.description")}
        showIcon
      />
      <div className="mt-4">
        <Button href="/ui/login">{t("pages.onboardingErrorView.backToLogin")}</Button>
      </div>
    </div>
  );
}
