import React from "react";
import { useTranslation } from "react-i18next";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Label } from "@/components/ui/label";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { buildDecisionCatalog, isDecisionMode, isDecisionSelection } from "@/lib/decisionModels";
import { defaultJevClassifierConfig } from "./jev_classifier_config";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

export default function DecisionModelSelect({
  value,
  onChange,
  modelInfo,
  deployments,
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelInfo: ModelGroup[];
  deployments: AutoRouterDeployment[] | undefined;
}) {
  const { t } = useTranslation();
  const id = React.useId();
  const { data: modelCostMapData } = useModelCostMap(true, true);
  const decisionCatalog = React.useMemo(() => buildDecisionCatalog(modelCostMapData), [modelCostMapData]);
  const decisionModels = modelInfo.filter(
    (model) =>
      isDecisionMode(model.mode) ||
      (deployments ?? []).some(
        (deployment) =>
          deployment.model_name === model.model_group &&
          isDecisionSelection(
            decisionCatalog,
            deployment.litellm_params?.custom_llm_provider ?? deployment.litellm_params?.model?.split("/")[0],
            [deployment.litellm_params?.model ?? "", deployment.litellm_params?.base_model ?? ""],
          ),
      ),
  );

  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{t("addModel.decisionModelSelect.label")}</Label>
      <SearchSelect
        inputId={id}
        aria-label={t("addModel.decisionModelSelect.label")}
        options={decisionModels.map((model) => ({ value: model.model_group, label: model.model_group }))}
        value={value.jev_classifier_config?.deployment_name ?? ""}
        placeholder={t("addModel.decisionModelSelect.placeholder")}
        allowClear={false}
        onValueChange={(deploymentName) => {
          if (!deploymentName) return;
          onChange({
            ...value,
            jev_classifier_config: {
              ...(value.jev_classifier_config ?? defaultJevClassifierConfig()),
              deployment_name: deploymentName,
            },
          });
        }}
      />
    </div>
  );
}
