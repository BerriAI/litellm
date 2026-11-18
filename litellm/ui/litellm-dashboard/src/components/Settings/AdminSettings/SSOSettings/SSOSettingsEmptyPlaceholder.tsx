import { Shield } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";

interface SSOSettingsEmptyPlaceholderProps {
  onAdd: () => void;
}

export default function SSOSettingsEmptyPlaceholder({ onAdd }: SSOSettingsEmptyPlaceholderProps) {
  const { t } = useTranslation();
  return (
    <div className="flex w-full flex-col items-center rounded-lg border border-dashed border-border bg-card p-12 text-center">
      <div className="mb-4 flex size-12 items-center justify-center rounded-full bg-muted">
        <Shield className="size-6 text-muted-foreground" />
      </div>
      <h4 className="text-base font-semibold text-foreground">{t("adminPanel.ssoSettings.emptyPlaceholder.title")}</h4>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
        {t("adminPanel.ssoSettings.emptyPlaceholder.description")}
      </p>
      <Button size="lg" onClick={onAdd} className="mt-4">
        {t("adminPanel.ssoSettings.emptyPlaceholder.configure")}
      </Button>
    </div>
  );
}
