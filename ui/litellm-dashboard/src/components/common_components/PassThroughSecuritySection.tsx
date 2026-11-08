import React from "react";
import { useTranslation } from "react-i18next";

import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";

export interface PassThroughSecuritySectionProps {
  authEnabled: boolean;
  onAuthChange: (checked: boolean) => void;
}

const PassThroughSecuritySection: React.FC<PassThroughSecuritySectionProps> = ({ authEnabled, onAuthChange }) => {
  const { t } = useTranslation();
  return (
    <Card className="block p-6">
      <h3 className="mb-2 text-lg font-semibold text-foreground">{t("passThroughSecuritySection.title")}</h3>
      <p className="mb-4 text-sm text-muted-foreground">{t("passThroughSecuritySection.description")}</p>
      <Switch
        aria-label={t("passThroughSecuritySection.requireAuth")}
        checked={authEnabled}
        onCheckedChange={onAuthChange}
      />
    </Card>
  );
};

export default PassThroughSecuritySection;
