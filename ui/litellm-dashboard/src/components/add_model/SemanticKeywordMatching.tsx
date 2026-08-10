import { InfoCircleOutlined } from "@ant-design/icons";
import { InputNumber, Select as AntdSelect, Switch, Tooltip, Typography } from "antd";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import { useTranslation } from "react-i18next";

const { Text } = Typography;

const DEFAULT_MATCH_THRESHOLD = 0.5;

interface SemanticKeywordMatchingProps {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  embeddingModel: string | undefined;
  onEmbeddingModelChange: (model: string) => void;
  matchThreshold: number;
  onMatchThresholdChange: (threshold: number) => void;
  modelInfo: ModelGroup[];
  showValidationErrors?: boolean;
}

const SemanticKeywordMatching: React.FC<SemanticKeywordMatchingProps> = ({
  enabled,
  onEnabledChange,
  embeddingModel,
  onEmbeddingModelChange,
  matchThreshold,
  onMatchThresholdChange,
  modelInfo,
  showValidationErrors = false,
}) => {
  const { t } = useTranslation("gateway");
  const embeddingModels = modelInfo.filter((model) => model.mode === "embedding");
  const modelOptions = Array.from(new Set(embeddingModels.map((model) => model.model_group))).map((model_group) => ({
    value: model_group,
    label: model_group,
  }));
  const embeddingModelMissing = showValidationErrors && !embeddingModel;

  return (
    <div>
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Text className="font-medium">{t("models.autoRouters.details.semantic.title")}</Text>
            <Tooltip title={t("models.autoRouters.details.semantic.tooltip")}>
              <InfoCircleOutlined className="text-gray-400" />
            </Tooltip>
          </div>
          <Text className="text-gray-500 text-sm">{t("models.autoRouters.details.semantic.description")}</Text>
        </div>
        <Switch
          checked={enabled}
          onChange={onEnabledChange}
          aria-label={t("models.autoRouters.details.semantic.title")}
        />
      </div>

      {enabled && (
        <div className="grid gap-4 md:grid-cols-2 mt-4 pt-4 border-t border-gray-200">
          <div>
            <Text className="text-sm font-medium mb-1 block">
              {t("models.autoRouters.details.semantic.embeddingModel")}
            </Text>
            <AntdSelect
              value={embeddingModel}
              onChange={onEmbeddingModelChange}
              placeholder={t("models.autoRouters.details.semantic.modelPlaceholder")}
              showSearch
              style={{ width: "100%" }}
              options={modelOptions}
              status={embeddingModelMissing ? "error" : undefined}
            />
            {embeddingModelMissing && (
              <Text type="danger" style={{ fontSize: 12 }}>
                {t("models.autoRouters.details.semantic.modelRequired")}
              </Text>
            )}
          </div>
          <div>
            <Text className="text-sm font-medium mb-1 block">
              {t("models.autoRouters.details.semantic.minimumScore")}
            </Text>
            <InputNumber
              value={matchThreshold}
              onChange={(value) => onMatchThresholdChange(value ?? DEFAULT_MATCH_THRESHOLD)}
              min={0}
              max={1}
              step={0.05}
              style={{ width: "100%" }}
            />
            <Text className="text-gray-500 text-xs mt-1 block">
              {t("models.autoRouters.details.semantic.scoreDescription")}
            </Text>
          </div>
        </div>
      )}
    </div>
  );
};

export default SemanticKeywordMatching;
export { DEFAULT_MATCH_THRESHOLD };
