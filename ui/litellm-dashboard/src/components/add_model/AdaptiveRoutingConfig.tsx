import { Card, InputNumber, Radio, Slider, Space, Switch, Typography } from "antd";
import React from "react";
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

  // AdaptiveRouterWeights requires quality + cost === 1.0, so a single quality
  // slider derives cost rather than exposing two independent inputs that could
  // drift apart and fail backend validation.
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
        <Switch checked={value.adaptive ?? false} onChange={handleAdaptiveToggle} />
        <Text strong>Enable adaptive bandit selection</Text>
      </div>
      <Text type="secondary" style={{ display: "block", fontSize: 12 }}>
        When disabled, each request always uses the model assigned to its classified tier.
      </Text>

      <Card className="bg-gray-50 mt-4">
        <Text strong style={{ display: "block", marginBottom: 8 }}>
          How Adaptive Routing Works
        </Text>
        <Text type="secondary" style={{ fontSize: 13 }}>
          It learns from how each conversation actually goes: does the user have to rephrase or correct the model, does
          it get stuck repeating itself, does it run out of tool calls, does the user seem satisfied. Combined with
          cost, this live feedback shifts future routing toward the models that are actually working well, and improves
          as more conversations come in. Until there&apos;s enough feedback, it defaults to the classified tier&apos;s
          model.
        </Text>
      </Card>

      {value.adaptive && (
        <div className="mt-4 space-y-4">
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              Quality vs. Cost ({Math.round(adaptiveWeights.quality * 100)}% quality /{" "}
              {Math.round(adaptiveWeights.cost * 100)}% cost)
            </Text>
            <Slider
              min={0}
              max={100}
              value={Math.round(adaptiveWeights.quality * 100)}
              onChange={handleQualityWeightChange}
              tooltip={{ formatter: (v) => `${v}% quality / ${100 - (v ?? 0)}% cost` }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              Higher quality weight favors more capable (pricier) models; higher cost weight favors cheaper models when
              the bandit has feedback to act on. Recommended: 30% quality / 70% cost split.
            </Text>
          </div>

          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              Eligible Model Pool
            </Text>
            <Radio.Group
              value={adaptiveEligible}
              onChange={(e) => handleAdaptiveEligibleChange(e.target.value)}
              className="w-full"
            >
              <Space direction="vertical" className="w-full">
                <Radio value="all">
                  <Text strong>All tiers (soft floor)</Text>{" "}
                  <Text type="secondary">— router can pick across tiers, depending on the best fit for the prompt</Text>
                </Radio>
                <Radio value="classified_tier">
                  <Text strong>Classified tier only</Text>{" "}
                  <Text type="secondary">— router can only pick models within tier</Text>
                </Radio>
              </Space>
            </Radio.Group>
          </div>

          {adaptiveEligible === "all" && (
            <div>
              <Text strong style={{ display: "block", marginBottom: 4 }}>
                Tier Distance Penalty
              </Text>
              <InputNumber
                value={tierDistancePenalty}
                onChange={handleTierDistancePenaltyChange}
                min={0}
                step={0.1}
                style={{ width: "100%" }}
              />
              <Text type="secondary" style={{ fontSize: 12 }}>
                Score penalty applied per tier-step away from the classified tier.
              </Text>
            </div>
          )}
        </div>
      )}
    </>
  );
};

export default AdaptiveRoutingConfig;
