import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Card, Divider, Space, Tooltip, Typography } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";
import { ModelGroup } from "@/components/llm_calls/fetch_models";

const { Text } = Typography;

interface ComplexityTiers {
  SIMPLE: string;
  MEDIUM: string;
  COMPLEX: string;
  REASONING: string;
}

interface ComplexityRouterConfigProps {
  modelInfo: ModelGroup[];
  value: ComplexityTiers;
  onChange: (tiers: ComplexityTiers) => void;
}

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({ modelInfo, value, onChange }) => {
  const { t } = useTranslation();

  const TIER_CONFIGS: Array<{
    key: keyof ComplexityTiers;
    labelKey: string;
    descriptionKey: string;
    examplesKey: string;
  }> = [
    {
      key: "SIMPLE",
      labelKey: "addModel.complexityRouterConfig.simpleTierLabel",
      descriptionKey: "addModel.complexityRouterConfig.simpleTierDescription",
      examplesKey: "addModel.complexityRouterConfig.simpleTierExamples",
    },
    {
      key: "MEDIUM",
      labelKey: "addModel.complexityRouterConfig.mediumTierLabel",
      descriptionKey: "addModel.complexityRouterConfig.mediumTierDescription",
      examplesKey: "addModel.complexityRouterConfig.mediumTierExamples",
    },
    {
      key: "COMPLEX",
      labelKey: "addModel.complexityRouterConfig.complexTierLabel",
      descriptionKey: "addModel.complexityRouterConfig.complexTierDescription",
      examplesKey: "addModel.complexityRouterConfig.complexTierExamples",
    },
    {
      key: "REASONING",
      labelKey: "addModel.complexityRouterConfig.reasoningTierLabel",
      descriptionKey: "addModel.complexityRouterConfig.reasoningTierDescription",
      examplesKey: "addModel.complexityRouterConfig.reasoningTierExamples",
    },
  ];

  // Prepare model options for dropdowns
  const modelOptions = modelInfo.map((model) => ({
    value: model.model_group,
    label: model.model_group,
  }));

  const handleTierChange = (tier: keyof ComplexityTiers, model: string) => {
    onChange({
      ...value,
      [tier]: model,
    });
  };

  return (
    <div className="w-full max-w-none">
      <Space align="center" style={{ marginBottom: 16 }}>
        <Typography.Title level={4} style={{ margin: 0 }}>
          {t("addModel.complexityRouterConfig.title")}
        </Typography.Title>
        <Tooltip title={t("addModel.complexityRouterConfig.titleTooltip")}>
          <InfoCircleOutlined className="text-gray-400" />
        </Tooltip>
      </Space>

      <Text type="secondary" style={{ display: "block", marginBottom: 24 }}>
        {t("addModel.complexityRouterConfig.subtitle")}
      </Text>

      <Card>
        {TIER_CONFIGS.map((tierConfig, index) => {
          const label = t(tierConfig.labelKey);
          const description = t(tierConfig.descriptionKey);
          const examples = t(tierConfig.examplesKey);
          return (
            <div key={tierConfig.key}>
              {index > 0 && <Divider style={{ margin: "16px 0" }} />}
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <Text strong style={{ fontSize: 16 }}>
                    {t("addModel.complexityRouterConfig.tierLabel", { label })}
                  </Text>
                  <Tooltip title={description}>
                    <InfoCircleOutlined className="text-gray-400" />
                  </Tooltip>
                </div>
                <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
                  {t("addModel.complexityRouterConfig.examplesLabel", { examples })}
                </Text>
                <AntdSelect
                  value={value[tierConfig.key]}
                  onChange={(model) => handleTierChange(tierConfig.key, model)}
                  placeholder={t("addModel.complexityRouterConfig.selectModelPlaceholder", {
                    label: label.toLowerCase(),
                  })}
                  showSearch
                  style={{ width: "100%" }}
                  options={modelOptions}
                />
              </div>
            </div>
          );
        })}
      </Card>

      <Divider />

      <Card className="bg-gray-50">
        <Text strong style={{ display: "block", marginBottom: 8 }}>
          {t("addModel.complexityRouterConfig.howClassificationWorksTitle")}
        </Text>
        <Text type="secondary" style={{ fontSize: 13 }}>
          {t("addModel.complexityRouterConfig.howClassificationWorksDesc")}
        </Text>
        <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20, fontSize: 13, color: "rgba(0, 0, 0, 0.45)" }}>
          <li>
            <strong>SIMPLE</strong>
            {t("addModel.complexityRouterConfig.scoreRangeSimple")}
          </li>
          <li>
            <strong>MEDIUM</strong>
            {t("addModel.complexityRouterConfig.scoreRangeMedium")}
          </li>
          <li>
            <strong>COMPLEX</strong>
            {t("addModel.complexityRouterConfig.scoreRangeComplex")}
          </li>
          <li>
            <strong>REASONING</strong>
            {t("addModel.complexityRouterConfig.scoreRangeReasoning")}
          </li>
        </ul>
      </Card>
    </div>
  );
};

export default ComplexityRouterConfig;
