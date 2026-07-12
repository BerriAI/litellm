import { InfoCircleOutlined } from "@ant-design/icons";
import { Card, InputNumber, Select as AntdSelect, Switch, Tooltip, Typography } from "antd";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";

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
  const embeddingModels = modelInfo.filter((model) => model.mode === "embedding");
  const modelOptions = Array.from(new Set(embeddingModels.map((model) => model.model_group))).map((model_group) => ({
    value: model_group,
    label: model_group,
  }));
  const embeddingModelMissing = showValidationErrors && !embeddingModel;

  return (
    <Card className="mb-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Text className="font-medium">Semantic keyword matching</Text>
            <Tooltip title="Recognize related phrasing beyond exact keyword matches by comparing embeddings instead of plain text. Overrides direct keyword matching">
              <InfoCircleOutlined className="text-gray-400" />
            </Tooltip>
          </div>
          <Text className="text-gray-500 text-sm">
            Uses same keyword-tier pairs as above and overrides direct keyword matching. Adds latency based on embedding
            model network request.
          </Text>
        </div>
        <Switch checked={enabled} onChange={onEnabledChange} aria-label="Semantic keyword matching" />
      </div>

      {enabled && (
        <div className="grid gap-4 md:grid-cols-2 mt-4 pt-4 border-t border-gray-200">
          <div>
            <Text className="text-sm font-medium mb-1 block">Embedding model</Text>
            <AntdSelect
              value={embeddingModel}
              onChange={onEmbeddingModelChange}
              placeholder="Select an embedding model"
              showSearch
              style={{ width: "100%" }}
              options={modelOptions}
              status={embeddingModelMissing ? "error" : undefined}
            />
            {embeddingModelMissing && (
              <Text type="danger" style={{ fontSize: 12 }}>
                An embedding model is required
              </Text>
            )}
          </div>
          <div>
            <Text className="text-sm font-medium mb-1 block">Minimum match score</Text>
            <InputNumber
              value={matchThreshold}
              onChange={(value) => onMatchThresholdChange(value ?? DEFAULT_MATCH_THRESHOLD)}
              min={0}
              max={1}
              step={0.05}
              style={{ width: "100%" }}
            />
            <Text className="text-gray-500 text-xs mt-1 block">Match only at or above this similarity score.</Text>
          </div>
        </div>
      )}
    </Card>
  );
};

export default SemanticKeywordMatching;
export { DEFAULT_MATCH_THRESHOLD };
