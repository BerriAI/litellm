import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  formatGuardrailMode,
  formatLoggingOnlyScope,
  loggingOnlyContinueFromParams,
  modeIncludesLoggingOnly,
} from "./guardrail_info_helpers";

type GuardrailModeParams = {
  mode?: unknown;
  default_on?: boolean;
  logging_only_scope?: string | null;
  logging_only_continue_on_input_failure?: boolean | null;
};

export const GuardrailModeCard: React.FC<{ litellmParams: GuardrailModeParams }> = ({ litellmParams }) => {
  const { t } = useLanguage();
  return (
    <Card className="block p-6">
      <p>{t("Mode")}</p>
      <div className="mt-2">
        <h3 className="text-lg font-medium">{formatGuardrailMode(litellmParams.mode) || "-"}</h3>
        <Badge variant={litellmParams.default_on ? "secondary" : "outline"}>
          {litellmParams.default_on ? t("Default On") : t("Default Off")}
        </Badge>
      </div>
      {modeIncludesLoggingOnly(litellmParams.mode) && (
        <div className="mt-4">
          <p>{t("Logging only scope")}</p>
          <h3 className="text-lg font-medium">{formatLoggingOnlyScope(litellmParams.logging_only_scope)}</h3>
          <p className="mt-2">{t("Continue after a flagged request")}</p>
          <h3 className="text-lg font-medium">{loggingOnlyContinueFromParams(litellmParams) ? t("On") : t("Off")}</h3>
        </div>
      )}
    </Card>
  );
};

export const GuardrailModeRows: React.FC<{ litellmParams: GuardrailModeParams }> = ({ litellmParams }) => {
  const { t } = useLanguage();
  return (
    <>
      <div>
        <p className="font-medium">{t("Mode")}</p>
        <div>{formatGuardrailMode(litellmParams.mode) || "-"}</div>
      </div>
      {modeIncludesLoggingOnly(litellmParams.mode) && (
        <div>
          <p className="font-medium">{t("Logging only scope")}</p>
          <div>{formatLoggingOnlyScope(litellmParams.logging_only_scope)}</div>
          <p className="font-medium">{t("Continue after a flagged request")}</p>
          <div>{loggingOnlyContinueFromParams(litellmParams) ? t("On") : t("Off")}</div>
        </div>
      )}
    </>
  );
};
