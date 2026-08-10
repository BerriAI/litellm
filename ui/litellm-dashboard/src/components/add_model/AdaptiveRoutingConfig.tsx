import { Card, InputNumber, Radio, Slider, Space, Switch, Typography } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";
import {
  AdaptiveEligible,
  ComplexityRouterConfigValue,
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "./ComplexityRouterConfig";

const { Text } = Typography;

interface AdaptiveRoutingConfigProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}

const AdaptiveRoutingConfig: React.FC<AdaptiveRoutingConfigProps> = ({ value, onChange }) => {
  const { t } = useTranslation("gateway");
  const adaptiveWeights = value.adaptive_weights ?? DEFAULT_ADAPTIVE_WEIGHTS;
  const adaptiveEligible = value.adaptive_eligible ?? "all";
  const tierDistancePenalty = value.tier_distance_penalty ?? DEFAULT_TIER_DISTANCE_PENALTY;

  const handleAdaptiveToggle = (adaptive: boolean) => {
    const nextValue: ComplexityRouterConfigValue = {
      ...value,
      adaptive,
      adaptive_weights: adaptiveWeights,
      adaptive_eligible: adaptiveEligible,
      tier_distance_penalty: tierDistancePenalty,
    };
    onChange(nextValue);
  };

  const handleQualityWeightChange = (qualityPercent: number) => {
    const quality = qualityPercent / 100;
    onChange({ ...value, adaptive_weights: { quality, cost: Math.round((1 - quality) * 100) / 100 } });
  };

  const handleAdaptiveEligibleChange = (eligible: AdaptiveEligible) => {
    onChange({ ...value, adaptive_eligible: eligible });
  };

  const handleTierDistancePenaltyChange = (penalty: number | null) => {
    onChange({ ...value, tier_distance_penalty: penalty ?? DEFAULT_TIER_DISTANCE_PENALTY });
  };

  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.adaptive ?? false}
          onChange={handleAdaptiveToggle}
          aria-label={t("models.autoRouters.details.adaptive.enable")}
        />
        <Text strong>{t("models.autoRouters.details.adaptive.enable")}</Text>
      </div>
      <Text type="secondary" style={{ display: "block", fontSize: 12 }}>
        {t("models.autoRouters.details.adaptive.disabledDescription")}
      </Text>

      <Card className="bg-gray-50 mt-4">
        <Text strong style={{ display: "block", marginBottom: 8 }}>
          {t("models.autoRouters.details.adaptive.howItWorks")}
        </Text>
        <Text type="secondary" style={{ fontSize: 13 }}>
          {t("models.autoRouters.details.adaptive.description")}
        </Text>
      </Card>

      {value.adaptive && (
        <div className="mt-4 space-y-4">
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.adaptive.qualityCost", {
                quality: Math.round(adaptiveWeights.quality * 100),
                cost: Math.round(adaptiveWeights.cost * 100),
              })}
            </Text>
            <Slider
              min={0}
              max={100}
              value={Math.round(adaptiveWeights.quality * 100)}
              onChange={handleQualityWeightChange}
              tooltip={{
                formatter: (v) =>
                  t("models.autoRouters.details.adaptive.tooltip", {
                    quality: v,
                    cost: 100 - (v ?? 0),
                  }),
              }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {t("models.autoRouters.details.adaptive.weightDescription")}
            </Text>
          </div>

          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.adaptive.pool")}
            </Text>
            <Radio.Group
              value={adaptiveEligible}
              onChange={(e) => handleAdaptiveEligibleChange(e.target.value)}
              className="w-full"
            >
              <Space direction="vertical" className="w-full">
                <Radio value="all">
                  <Text strong>{t("models.autoRouters.details.adaptive.allTiers")}</Text>{" "}
                  <Text type="secondary">{t("models.autoRouters.details.adaptive.allTiersDescription")}</Text>
                </Radio>
                <Radio value="classified_tier">
                  <Text strong>{t("models.autoRouters.details.adaptive.classifiedTier")}</Text>{" "}
                  <Text type="secondary">{t("models.autoRouters.details.adaptive.classifiedTierDescription")}</Text>
                </Radio>
              </Space>
            </Radio.Group>
          </div>

          {adaptiveEligible === "all" && (
            <div>
              <Text strong style={{ display: "block", marginBottom: 4 }}>
                {t("models.autoRouters.details.adaptive.distancePenalty")}
              </Text>
              <InputNumber
                value={tierDistancePenalty}
                onChange={handleTierDistancePenaltyChange}
                min={0}
                step={0.1}
                style={{ width: "100%" }}
              />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {t("models.autoRouters.details.adaptive.distancePenaltyDescription")}
              </Text>
            </div>
          )}
        </div>
      )}
    </>
  );
};

export default AdaptiveRoutingConfig;
