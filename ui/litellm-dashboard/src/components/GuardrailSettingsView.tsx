import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { Globe2 } from "lucide-react";
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
  const { t } = useLanguage();
  const optedOutSet = new Set(optedOutGlobalGuardrails);
  const globalsRunning = Array.from(globalGuardrailNames).filter((n) => !optedOutSet.has(n));
  const nonGlobalOptIns = teamGuardrails.filter((n) => !globalGuardrailNames.has(n));

  const isEmpty = !killSwitchOn && globalsRunning.length === 0 && nonGlobalOptIns.length === 0;

  const content = isEmpty ? (
    <span className="block text-muted-foreground">{t("No guardrails configured")}</span>
  ) : (
    <div className="flex flex-col gap-4">
      <div>
        <span className="mb-2 flex items-center gap-1 text-sm font-medium text-foreground">
          <Globe2 className="size-4" aria-label={t("Global guardrail")} />
          {t("Global")}
        </span>
        {killSwitchOn ? (
          <Badge variant="outline">{t("Bypassed for this team")}</Badge>
        ) : globalsRunning.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {globalsRunning.map((name) => (
              <Badge key={name}>{name}</Badge>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-muted-foreground">{t("None configured")}</span>
        )}
      </div>
      <div>
        <span className="mb-2 block text-sm font-medium text-foreground">{t("Team-specific")}</span>
        {nonGlobalOptIns.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {nonGlobalOptIns.map((name) => (
              <Badge key={name}>{name}</Badge>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-muted-foreground">{t("None configured")}</span>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{t("Guardrails Settings")}</CardTitle>
          <CardDescription>{t("Global and team-specific guardrails applied to this team")}</CardDescription>
        </CardHeader>
        <CardContent>{content}</CardContent>
      </Card>
    );
  }

  return (
    <div className={cn(className)}>
      <span className="mb-3 block font-medium text-foreground">{t("Guardrails Settings")}</span>
      {content}
    </div>
  );
}

export default GuardrailSettingsView;
