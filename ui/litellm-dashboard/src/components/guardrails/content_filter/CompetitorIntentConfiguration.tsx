import React, { useEffect, useState } from "react";
import { Card, Typography, Select, Switch, Form, Space, InputNumber } from "antd";
import { getMajorAirlines } from "../../networking";
import { useTranslation } from "react-i18next";

const { Title, Text } = Typography;
const { Option } = Select;

export interface MajorAirline {
  id: string;
  match: string;
  tags: string[];
}

export interface CompetitorIntentConfig {
  competitor_intent_type: "airline" | "generic";
  brand_self: string[];
  locations?: string[];
  competitors?: string[];
  policy?: {
    competitor_comparison?: "refuse" | "reframe";
    possible_competitor_comparison?: "refuse" | "reframe";
  };
  threshold_high?: number;
  threshold_medium?: number;
  threshold_low?: number;
}

interface CompetitorIntentConfigurationProps {
  enabled: boolean;
  config: CompetitorIntentConfig | null;
  onChange: (enabled: boolean, config: CompetitorIntentConfig | null) => void;
  accessToken?: string | null;
}

const DEFAULT_CONFIG: CompetitorIntentConfig = {
  competitor_intent_type: "airline",
  brand_self: [],
  locations: [],
  policy: {
    competitor_comparison: "refuse",
    possible_competitor_comparison: "reframe",
  },
  threshold_high: 0.7,
  threshold_medium: 0.45,
  threshold_low: 0.3,
};

const CompetitorIntentConfiguration: React.FC<CompetitorIntentConfigurationProps> = ({
  enabled,
  config,
  onChange,
  accessToken,
}) => {
  const { t } = useTranslation();
  const effectiveConfig = config ?? DEFAULT_CONFIG;
  const [airlineOptions, setAirlineOptions] = useState<MajorAirline[]>([]);
  const [loadingAirlines, setLoadingAirlines] = useState(false);

  useEffect(() => {
    if (effectiveConfig.competitor_intent_type === "airline" && accessToken && airlineOptions.length === 0) {
      setLoadingAirlines(true);
      getMajorAirlines(accessToken)
        .then((res) => setAirlineOptions(res.airlines ?? []))
        .catch(() => setAirlineOptions([]))
        .finally(() => setLoadingAirlines(false));
    }
  }, [effectiveConfig.competitor_intent_type, accessToken, airlineOptions.length]);

  const handleEnabledChange = (checked: boolean) => {
    onChange(checked, checked ? { ...DEFAULT_CONFIG } : null);
  };

  const handleConfigChange = (field: string, value: unknown) => {
    onChange(enabled, { ...effectiveConfig, [field]: value });
  };

  const handlePolicyChange = (key: string, value: string) => {
    onChange(enabled, {
      ...effectiveConfig,
      policy: { ...effectiveConfig.policy, [key]: value },
    });
  };

  const handleNestedArrayChange = (field: "brand_self" | "locations" | "competitors", values: string[]) => {
    onChange(enabled, { ...effectiveConfig, [field]: values.filter(Boolean) });
  };

  const handleBrandSelfChange = (values: string[]) => {
    const filtered = values.filter(Boolean);
    const expanded: string[] = [];
    const seen = new Set<string>();
    for (const v of filtered) {
      const airline = airlineOptions.find((a) => {
        const primary = a.match.split("|")[0]?.trim().toLowerCase();
        return primary === v.toLowerCase();
      });
      if (airline) {
        for (const variant of airline.match
          .split("|")
          .map((s) => s.trim().toLowerCase())
          .filter(Boolean)) {
          if (!seen.has(variant)) {
            seen.add(variant);
            expanded.push(variant);
          }
        }
      } else if (!seen.has(v.toLowerCase())) {
        seen.add(v.toLowerCase());
        expanded.push(v);
      }
    }
    onChange(enabled, { ...effectiveConfig, brand_self: expanded });
  };

  if (!enabled) {
    return (
      <Card
        title={
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Title level={5} style={{ margin: 0 }}>
              {t("guardrails.competitorIntentConfiguration.title")}
            </Title>
            <Switch checked={false} onChange={handleEnabledChange} />
          </div>
        }
        size="small"
      >
        <Text type="secondary">{t("guardrails.competitorIntentConfiguration.disabledDescription")}</Text>
      </Card>
    );
  }

  return (
    <Card
      title={
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Title level={5} style={{ margin: 0 }}>
            {t("guardrails.competitorIntentConfiguration.title")}
          </Title>
          <Switch checked={enabled} onChange={handleEnabledChange} />
        </div>
      }
      size="small"
    >
      <Text type="secondary" style={{ display: "block", marginBottom: 16 }}>
        {t("guardrails.competitorIntentConfiguration.enabledDescription")}
      </Text>
      <Form layout="vertical" size="small">
        <Form.Item label={t("common.type")}>
          <Select
            value={effectiveConfig.competitor_intent_type}
            onChange={(v) => handleConfigChange("competitor_intent_type", v)}
            style={{ width: "100%" }}
          >
            <Option value="airline">{t("guardrails.competitorIntentConfiguration.typeAirline")}</Option>
            <Option value="generic">{t("guardrails.competitorIntentConfiguration.typeGeneric")}</Option>
          </Select>
        </Form.Item>

        <Form.Item
          label={t("guardrails.competitorIntentConfiguration.brandSelfLabel")}
          required
          help={
            effectiveConfig.competitor_intent_type === "airline"
              ? t("guardrails.competitorIntentConfiguration.brandSelfHelpAirline")
              : t("guardrails.competitorIntentConfiguration.brandSelfHelpGeneric")
          }
        >
          <Select
            mode="tags"
            style={{ width: "100%" }}
            placeholder={
              loadingAirlines
                ? t("guardrails.competitorIntentConfiguration.loadingAirlines")
                : effectiveConfig.competitor_intent_type === "airline"
                  ? t("guardrails.competitorIntentConfiguration.brandSelfPlaceholderAirline")
                  : t("guardrails.competitorIntentConfiguration.brandSelfPlaceholderGeneric")
            }
            value={effectiveConfig.brand_self}
            onChange={(v) =>
              effectiveConfig.competitor_intent_type === "airline" && airlineOptions.length > 0
                ? handleBrandSelfChange(v ?? [])
                : handleNestedArrayChange("brand_self", v ?? [])
            }
            tokenSeparators={[","]}
            loading={loadingAirlines}
            showSearch
            filterOption={(input, option) =>
              (option?.label?.toString().toLowerCase() ?? "").includes(input.toLowerCase())
            }
            optionFilterProp="label"
            options={
              effectiveConfig.competitor_intent_type === "airline" && airlineOptions.length > 0
                ? airlineOptions.map((a) => {
                    const primary = a.match.split("|")[0]?.trim() ?? a.id;
                    const variants = a.match
                      .split("|")
                      .map((s) => s.trim().toLowerCase())
                      .filter(Boolean);
                    return {
                      value: primary.toLowerCase(),
                      label: `${primary}${variants.length > 1 ? ` (${variants.slice(1).join(", ")})` : ""}`,
                    };
                  })
                : undefined
            }
          />
        </Form.Item>

        {effectiveConfig.competitor_intent_type === "airline" && (
          <Form.Item
            label={t("guardrails.competitorIntentConfiguration.locationsLabel")}
            help={t("guardrails.competitorIntentConfiguration.locationsHelp")}
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              placeholder={t("guardrails.competitorIntentConfiguration.tagsPlaceholder")}
              value={effectiveConfig.locations ?? []}
              onChange={(v) => handleNestedArrayChange("locations", v ?? [])}
              tokenSeparators={[","]}
            />
          </Form.Item>
        )}

        {effectiveConfig.competitor_intent_type === "generic" && (
          <Form.Item
            label={t("guardrails.competitorIntentConfiguration.competitorsLabel")}
            required
            help={t("guardrails.competitorIntentConfiguration.competitorsHelp")}
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              placeholder={t("guardrails.competitorIntentConfiguration.tagsPlaceholder")}
              value={effectiveConfig.competitors ?? []}
              onChange={(v) => handleNestedArrayChange("competitors", v ?? [])}
              tokenSeparators={[","]}
            />
          </Form.Item>
        )}

        <Form.Item label={t("guardrails.competitorIntentConfiguration.policyCompetitorLabel")}>
          <Select
            value={effectiveConfig.policy?.competitor_comparison ?? "refuse"}
            onChange={(v) => handlePolicyChange("competitor_comparison", v)}
            style={{ width: "100%" }}
          >
            <Option value="refuse">{t("guardrails.competitorIntentConfiguration.policyRefuse")}</Option>
            <Option value="reframe">{t("guardrails.competitorIntentConfiguration.policyReframe")}</Option>
          </Select>
        </Form.Item>

        <Form.Item label={t("guardrails.competitorIntentConfiguration.policyPossibleCompetitorLabel")}>
          <Select
            value={effectiveConfig.policy?.possible_competitor_comparison ?? "reframe"}
            onChange={(v) => handlePolicyChange("possible_competitor_comparison", v)}
            style={{ width: "100%" }}
          >
            <Option value="refuse">{t("guardrails.competitorIntentConfiguration.policyRefuse")}</Option>
            <Option value="reframe">{t("guardrails.competitorIntentConfiguration.policyReframeBackend")}</Option>
          </Select>
        </Form.Item>

        <Form.Item
          label={t("guardrails.competitorIntentConfiguration.confidenceThresholdsLabel")}
          help={
            <>
              {t("guardrails.competitorIntentConfiguration.confidenceThresholdsHelp")}
              <ul style={{ marginBottom: 0, marginTop: 4, paddingLeft: 20 }}>
                <li>
                  <strong>{t("guardrails.competitorIntentConfiguration.thresholdHighLabel")}</strong>:{" "}
                  {t("guardrails.competitorIntentConfiguration.thresholdHighDesc")}
                </li>
                <li>
                  <strong>{t("guardrails.competitorIntentConfiguration.thresholdMediumLabel")}</strong>:{" "}
                  {t("guardrails.competitorIntentConfiguration.thresholdMediumDesc")}
                </li>
                <li>
                  <strong>{t("guardrails.competitorIntentConfiguration.thresholdLowLabel")}</strong>:{" "}
                  {t("guardrails.competitorIntentConfiguration.thresholdLowDesc")}
                </li>
              </ul>
              {t("guardrails.competitorIntentConfiguration.thresholdsNote")}
            </>
          }
        >
          <Space wrap>
            <Form.Item
              label={t("guardrails.competitorIntentConfiguration.thresholdHighShort")}
              style={{ marginBottom: 0 }}
              help={t("guardrails.competitorIntentConfiguration.thresholdHighExample")}
            >
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={effectiveConfig.threshold_high ?? 0.7}
                onChange={(v) => handleConfigChange("threshold_high", v ?? 0.7)}
                style={{ width: 80 }}
              />
            </Form.Item>
            <Form.Item
              label={t("guardrails.competitorIntentConfiguration.thresholdMediumShort")}
              style={{ marginBottom: 0 }}
              help={t("guardrails.competitorIntentConfiguration.thresholdMediumExample")}
            >
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={effectiveConfig.threshold_medium ?? 0.45}
                onChange={(v) => handleConfigChange("threshold_medium", v ?? 0.45)}
                style={{ width: 80 }}
              />
            </Form.Item>
            <Form.Item
              label={t("guardrails.competitorIntentConfiguration.thresholdLowShort")}
              style={{ marginBottom: 0 }}
              help={t("guardrails.competitorIntentConfiguration.thresholdLowExample")}
            >
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={effectiveConfig.threshold_low ?? 0.3}
                onChange={(v) => handleConfigChange("threshold_low", v ?? 0.3)}
                style={{ width: 80 }}
              />
            </Form.Item>
          </Space>
        </Form.Item>
      </Form>
    </Card>
  );
};

export default CompetitorIntentConfiguration;
