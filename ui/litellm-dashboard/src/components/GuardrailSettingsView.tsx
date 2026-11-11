import React from "react";
import { Globe2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cva.config";

interface GuardrailSettingsViewProps {
  globalGuardrailNames: Set<string>;
  teamGuardrails?: string[];
  optedOutGlobalGuardrails?: string[];
  killSwitchOn?: boolean;
  variant?: "card" | "inline";
  className?: string;
}

export function GuardrailSettingsView({
  globalGuardrailNames,
  teamGuardrails = [],
  optedOutGlobalGuardrails = [],
  killSwitchOn = false,
  variant = "card",
  className = "",
}: GuardrailSettingsViewProps) {
  const { t } = useTranslation();
  const optedOutSet = new Set(optedOutGlobalGuardrails);
  const globalsRunning = Array.from(globalGuardrailNames).filter((n) => !optedOutSet.has(n));
  const nonGlobalOptIns = teamGuardrails.filter((n) => !globalGuardrailNames.has(n));

  const isEmpty = !killSwitchOn && globalsRunning.length === 0 && nonGlobalOptIns.length === 0;

  const content = isEmpty ? (
    <span className="block text-muted-foreground">{t("guardrailSettingsView.empty")}</span>
  ) : (
    <div className="flex flex-col gap-4">
      <div>
        <span className="mb-2 flex items-center gap-1 text-sm font-medium text-foreground">
          <Globe2 className="size-4" aria-label={t("guardrailSettingsView.globalAriaLabel")} />
          {t("guardrailSettingsView.global")}
        </span>
        {killSwitchOn ? (
          <Badge variant="outline">{t("guardrailSettingsView.bypassed")}</Badge>
        ) : globalsRunning.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {globalsRunning.map((name) => (
              <Badge key={name}>{name}</Badge>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-muted-foreground">{t("guardrailSettingsView.noneConfigured")}</span>
        )}
      </div>
      <div>
        <span className="mb-2 block text-sm font-medium text-foreground">
          {t("guardrailSettingsView.teamSpecific")}
        </span>
        {nonGlobalOptIns.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {nonGlobalOptIns.map((name) => (
              <Badge key={name}>{name}</Badge>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-muted-foreground">{t("guardrailSettingsView.noneConfigured")}</span>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{t("guardrailSettingsView.title")}</CardTitle>
          <CardDescription>{t("guardrailSettingsView.description")}</CardDescription>
        </CardHeader>
        <CardContent>{content}</CardContent>
      </Card>
    );
  }

  return (
    <div className={cn(className)}>
      <span className="mb-3 block font-medium text-foreground">{t("guardrailSettingsView.title")}</span>
      {content}
    </div>
  );
}

export default GuardrailSettingsView;
