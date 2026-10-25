import { useLanguage } from "@/contexts/LanguageContext";
import { Badge } from "@/components/ui/badge";
import { DEFAULT_DECISION_THRESHOLD } from "./decision_model/decisionModelQuestion";
import { GuardrailModeRows } from "./GuardrailModeDisplay";
import { GuardrailStreamScopeDetail } from "./StreamScopeFields";
import ToolPermissionRulesEditor, { type ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";

interface DecisionModelCheckView {
  name: string;
  action?: string;
  threshold?: number;
  instructions?: string;
}

const DecisionModelSection = ({
  decisionModel,
  checks,
}: {
  decisionModel: string;
  checks: DecisionModelCheckView[];
}) => {
  const { t } = useLanguage();
  return (
    <div>
      <p className="font-medium">{t("Decision Model")}</p>
      <div className="font-mono">{decisionModel}</div>
      <p className="mt-2 font-medium">{t("Checks")}</p>
      <div className="mt-2 overflow-hidden rounded-lg border border-border">
        <div className="flex border-b border-border bg-muted/40 px-4 py-2 text-xs font-semibold">
          <span className="flex-1">{t("Check")}</span>
          <span className="w-24 text-right">{t("Action")}</span>
          <span className="w-24 text-right">{t("Threshold")}</span>
        </div>
        {checks.map((check) => (
          <div key={check.name} className="flex items-center border-b border-border px-4 py-2 text-sm last:border-b-0">
            <span className="flex-1">{check.name.replace(/_/g, " ")}</span>
            <span className="w-24 text-right">
              <Badge variant="secondary">{check.action ?? "block"}</Badge>
            </span>
            <span className="w-24 text-right">{check.threshold ?? DEFAULT_DECISION_THRESHOLD}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export const GuardrailReadOnlyDetails = ({
  guardrailId,
  guardrailName,
  displayName,
  litellmParams,
  streamScope,
  defaultOn,
  piiEntityCount,
  createdAt,
  updatedAt,
  showToolPermission,
  toolPermissionConfig,
}: {
  guardrailId: string;
  guardrailName: string;
  displayName: string;
  litellmParams: {
    mode?: unknown;
    logging_only_scope?: string | null;
    guardrail?: string;
    decision_model?: string;
    checks?: DecisionModelCheckView[];
    logging_only_continue_on_input_failure?: boolean | null;
  };
  streamScope: unknown;
  defaultOn: boolean | undefined;
  piiEntityCount: number;
  createdAt: string;
  updatedAt: string;
  showToolPermission: boolean;
  toolPermissionConfig: ToolPermissionConfig;
}) => {
  const { t } = useLanguage();
  return (
    <div className="space-y-4">
      <div>
        <p className="font-medium">{t("Guardrail ID")}</p>
        <div className="font-mono">{guardrailId}</div>
      </div>
      <div>
        <p className="font-medium">{t("Guardrail Name")}</p>
        <div>{guardrailName || t("Unnamed Guardrail")}</div>
      </div>
      <div>
        <p className="font-medium">{t("Provider")}</p>
        <div>{displayName}</div>
      </div>
      <GuardrailModeRows litellmParams={litellmParams} />
      <GuardrailStreamScopeDetail raw={streamScope} />
      <div>
        <p className="font-medium">{t("Default On")}</p>
        <Badge variant={defaultOn ? "secondary" : "outline"}>{defaultOn ? t("Yes") : t("No")}</Badge>
      </div>
      {litellmParams.guardrail === "decision_model" && litellmParams.decision_model && (
        <DecisionModelSection decisionModel={litellmParams.decision_model} checks={litellmParams.checks ?? []} />
      )}
      {piiEntityCount > 0 && (
        <div>
          <p className="font-medium">{t("PII Protection")}</p>
          <div className="mt-2">
            <Badge variant="secondary">
              {piiEntityCount} {t("PII entities configured")}
            </Badge>
          </div>
        </div>
      )}
      <div>
        <p className="font-medium">{t("Created At")}</p>
        <div>{createdAt}</div>
      </div>
      <div>
        <p className="font-medium">{t("Last Updated")}</p>
        <div>{updatedAt}</div>
      </div>
      {showToolPermission && <ToolPermissionRulesEditor value={toolPermissionConfig} disabled />}
    </div>
  );
};
