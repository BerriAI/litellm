import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";

import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";

export interface PassThroughSecuritySectionProps {
  authEnabled: boolean;
  onAuthChange: (checked: boolean) => void;
}

const PassThroughSecuritySection: React.FC<PassThroughSecuritySectionProps> = ({ authEnabled, onAuthChange }) => {
  const { t } = useLanguage();
  return (
    <Card className="block p-6">
      <h3 className="mb-2 text-lg font-semibold text-foreground">{t("Security")}</h3>
      <p className="mb-4 text-sm text-muted-foreground">
        {t("When enabled, requests to this endpoint will require a valid LiteLLM Virtual Key")}
      </p>
      <Switch aria-label={t("Require LiteLLM authentication")} checked={authEnabled} onCheckedChange={onAuthChange} />
    </Card>
  );
};

export default PassThroughSecuritySection;
