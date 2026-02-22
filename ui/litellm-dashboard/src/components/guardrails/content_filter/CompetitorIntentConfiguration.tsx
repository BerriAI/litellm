import React, { useEffect, useState } from "react";
import {
  Card,
  Typography,
  Select,
  Switch,
  Form,
  Space,
  InputNumber,
} from "antd";
import { getMajorAirlines } from "../../networking";

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

const CompetitorIntentConfiguration: React.FC<
  CompetitorIntentConfigurationProps
> = ({ enabled, config, onChange, accessToken }) => {
  const effectiveConfig = config ?? DEFAULT_CONFIG;
  const [airlineOptions, setAirlineOptions] = useState<MajorAirline[]>([]);
  const [loadingAirlines, setLoadingAirlines] = useState(false);

  useEffect(() => {
    if (
      effectiveConfig.competitor_intent_type === "airline" &&
      accessToken &&
      airlineOptions.length === 0
    ) {
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
        for (const variant of airline.match.split("|").map((s) => s.trim().toLowerCase()).filter(Boolean)) {
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
              Competitor Intent Filter
            </Title>
            <Switch checked={false} onChange={handleEnabledChange} />
          </div>
        }
        size="small"
      >
        <Text type="secondary">
          Block or reframe competitor comparison questions. When enabled, airline type
          auto-loads competitors from IATA; generic type requires manual competitor list.
        </Text>
      </Card>
    );
  }

  return (
    <Card
      title={
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Title level={5} style={{ margin: 0 }}>
            Competitor Intent Filter
          </Title>
          <Switch checked={enabled} onChange={handleEnabledChange} />
        </div>
      }
      size="small"
    >
      <Text type="secondary" style={{ display: "block", marginBottom: 16 }}>
        Block or reframe competitor comparison questions. Airline type uses major airlines
        (excluding your brand); generic requires manual competitor list.
      </Text>
      <Form layout="vertical" size="small">
        <Form.Item label="Type">
          <Select
            value={effectiveConfig.competitor_intent_type}
            onChange={(v) => handleConfigChange("competitor_intent_type", v)}
            style={{ width: "100%" }}
          >
            <Option value="airline">Airline (auto-load competitors from IATA)</Option>
            <Option value="generic">Generic (specify competitors manually)</Option>
          </Select>
        </Form.Item>

        <Form.Item
          label="Your Brand (brand_self)"
          required
          help={
            effectiveConfig.competitor_intent_type === "airline"
              ? "Select your airline from the list (excluded from competitors) or type to add a custom term"
              : "Names/codes users use for your brand"
          }
        >
          <Select
            mode="tags"
            style={{ width: "100%" }}
            placeholder={
              loadingAirlines
                ? "Loading airlines..."
                : effectiveConfig.competitor_intent_type === "airline"
                  ? "Search or select airline, or type to add custom"
                  : "Type and press Enter to add"
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
              (option?.label?.toString().toLowerCase() ?? "").includes(
                input.toLowerCase()
              )
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
            label="Locations (optional)"
            help="Countries, cities, airports for disambiguation (e.g. qatar, doha)"
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              placeholder="Type and press Enter to add"
              value={effectiveConfig.locations ?? []}
              onChange={(v) => handleNestedArrayChange("locations", v ?? [])}
              tokenSeparators={[","]}
            />
          </Form.Item>
        )}

        {effectiveConfig.competitor_intent_type === "generic" && (
          <Form.Item
            label="Competitors"
            required
            help="Competitor names to detect (required for generic type)"
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              placeholder="Type and press Enter to add"
              value={effectiveConfig.competitors ?? []}
              onChange={(v) => handleNestedArrayChange("competitors", v ?? [])}
              tokenSeparators={[","]}
            />
          </Form.Item>
        )}

        <Form.Item label="Policy: Competitor comparison">
          <Select
            value={effectiveConfig.policy?.competitor_comparison ?? "refuse"}
            onChange={(v) => handlePolicyChange("competitor_comparison", v)}
            style={{ width: "100%" }}
          >
            <Option value="refuse">Refuse (block request)</Option>
            <Option value="reframe">Reframe (suggest alternative)</Option>
          </Select>
        </Form.Item>

        <Form.Item label="Policy: Possible competitor comparison">
          <Select
            value={effectiveConfig.policy?.possible_competitor_comparison ?? "reframe"}
            onChange={(v) => handlePolicyChange("possible_competitor_comparison", v)}
            style={{ width: "100%" }}
          >
            <Option value="refuse">Refuse (block request)</Option>
            <Option value="reframe">Reframe (suggest alternative to backend LLM)</Option>
          </Select>
        </Form.Item>

        <Form.Item
          label="Confidence thresholds"
          help={
            <>
              Classify competitor intent by confidence (0–1). Higher confidence → stronger intent.
              <ul style={{ marginBottom: 0, marginTop: 4, paddingLeft: 20 }}>
                <li>
                  <strong>High (≥)</strong>: Treat as full competitor comparison → uses &quot;Competitor comparison&quot; policy
                </li>
                <li>
                  <strong>Medium (≥)</strong>: Treat as possible comparison → uses &quot;Possible competitor comparison&quot; policy
                </li>
                <li>
                  <strong>Low (≥)</strong>: Log only; allow request. Below Low → allow with no action
                </li>
              </ul>
              Raise thresholds to be more permissive; lower them to be stricter.
            </>
          }
        >
          <Space wrap>
            <Form.Item label="High" style={{ marginBottom: 0 }} help="e.g. 0.7">
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={effectiveConfig.threshold_high ?? 0.7}
                onChange={(v) => handleConfigChange("threshold_high", v ?? 0.7)}
                style={{ width: 80 }}
              />
            </Form.Item>
            <Form.Item label="Medium" style={{ marginBottom: 0 }} help="e.g. 0.45">
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={effectiveConfig.threshold_medium ?? 0.45}
                onChange={(v) => handleConfigChange("threshold_medium", v ?? 0.45)}
                style={{ width: 80 }}
              />
            </Form.Item>
            <Form.Item label="Low" style={{ marginBottom: 0 }} help="e.g. 0.3">
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
