import type { ParseKeys } from "i18next";
import React, { useEffect, useId, useState } from "react";
import { Trans, useTranslation } from "react-i18next";

import { getMajorAirlines } from "@/components/networking";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";

import { TagsInput } from "./TagsInput";
import { ThresholdInput } from "./ThresholdInput";

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

const INTENT_TYPES: readonly { value: CompetitorIntentConfig["competitor_intent_type"]; labelKey: ParseKeys }[] = [
  { value: "airline", labelKey: "guardrails.contentFilter.competitorIntent.intentTypes.airline" },
  { value: "generic", labelKey: "guardrails.contentFilter.competitorIntent.intentTypes.generic" },
];

type PolicyChoice = "refuse" | "reframe";

const COMPETITOR_COMPARISON_POLICIES: readonly { value: PolicyChoice; labelKey: ParseKeys }[] = [
  { value: "refuse", labelKey: "guardrails.contentFilter.competitorIntent.policies.refuse" },
  { value: "reframe", labelKey: "guardrails.contentFilter.competitorIntent.policies.reframe" },
];

const POSSIBLE_COMPETITOR_COMPARISON_POLICIES: readonly { value: PolicyChoice; labelKey: ParseKeys }[] = [
  { value: "refuse", labelKey: "guardrails.contentFilter.competitorIntent.policies.refuse" },
  { value: "reframe", labelKey: "guardrails.contentFilter.competitorIntent.policies.reframeToBackend" },
];

const THRESHOLDS = [
  {
    field: "threshold_high",
    labelKey: "guardrails.contentFilter.competitorIntent.thresholds.high",
    hint: 0.7,
    fallback: 0.7,
  },
  {
    field: "threshold_medium",
    labelKey: "guardrails.contentFilter.competitorIntent.thresholds.medium",
    hint: 0.45,
    fallback: 0.45,
  },
  {
    field: "threshold_low",
    labelKey: "guardrails.contentFilter.competitorIntent.thresholds.low",
    hint: 0.3,
    fallback: 0.3,
  },
] as const satisfies readonly { field: string; labelKey: ParseKeys; hint: number; fallback: number }[];

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
  const fieldId = useId();

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

  const header = (
    <CardHeader className="gap-0">
      <CardTitle className="text-base">{t("guardrails.contentFilter.competitorIntent.title")}</CardTitle>
      <CardAction>
        <Switch checked={enabled} onCheckedChange={handleEnabledChange} />
      </CardAction>
    </CardHeader>
  );

  if (!enabled) {
    return (
      <Card>
        {header}
        <CardContent>
          <p className="text-sm text-muted-foreground">
            {t("guardrails.contentFilter.competitorIntent.disabledDescription")}
          </p>
        </CardContent>
      </Card>
    );
  }

  const airlineTags =
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
      : [];

  const intentTypes = INTENT_TYPES.map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
  const competitorComparisonPolicies = COMPETITOR_COMPARISON_POLICIES.map(({ value, labelKey }) => ({
    value,
    label: t(labelKey),
  }));
  const possibleCompetitorComparisonPolicies = POSSIBLE_COMPETITOR_COMPARISON_POLICIES.map(({ value, labelKey }) => ({
    value,
    label: t(labelKey),
  }));

  return (
    <Card>
      {header}
      <CardContent>
        <p className="mb-4 text-sm text-muted-foreground">
          {t("guardrails.contentFilter.competitorIntent.enabledDescription")}
        </p>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor={`${fieldId}-type`}>
              {t("guardrails.contentFilter.competitorIntent.typeLabel")}
            </FieldLabel>
            <Select
              items={intentTypes}
              value={effectiveConfig.competitor_intent_type}
              onValueChange={(v: string | null) => v !== null && handleConfigChange("competitor_intent_type", v)}
            >
              <SelectTrigger id={`${fieldId}-type`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {intentTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value} title={type.label}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field>
            <FieldLabel htmlFor={`${fieldId}-brand-self`}>
              {t("guardrails.contentFilter.competitorIntent.brandSelfLabel", { field: "brand_self" })}
            </FieldLabel>
            <TagsInput
              id={`${fieldId}-brand-self`}
              value={effectiveConfig.brand_self}
              onValueChange={(v) =>
                effectiveConfig.competitor_intent_type === "airline" && airlineOptions.length > 0
                  ? handleBrandSelfChange(v)
                  : handleNestedArrayChange("brand_self", v)
              }
              options={airlineTags}
              tokenSeparators={[","]}
              loading={loadingAirlines}
              placeholder={
                effectiveConfig.competitor_intent_type === "airline"
                  ? t("guardrails.contentFilter.competitorIntent.airlinePlaceholder")
                  : t("guardrails.contentFilter.competitorIntent.typeToAddPlaceholder")
              }
            />
            <FieldDescription>
              {effectiveConfig.competitor_intent_type === "airline"
                ? t("guardrails.contentFilter.competitorIntent.airlineBrandDescription")
                : t("guardrails.contentFilter.competitorIntent.genericBrandDescription")}
            </FieldDescription>
          </Field>

          {effectiveConfig.competitor_intent_type === "airline" && (
            <Field>
              <FieldLabel htmlFor={`${fieldId}-locations`}>
                {t("guardrails.contentFilter.competitorIntent.locationsLabel")}
              </FieldLabel>
              <TagsInput
                id={`${fieldId}-locations`}
                value={effectiveConfig.locations ?? []}
                onValueChange={(v) => handleNestedArrayChange("locations", v)}
                tokenSeparators={[","]}
                placeholder={t("guardrails.contentFilter.competitorIntent.typeToAddPlaceholder")}
              />
              <FieldDescription>{t("guardrails.contentFilter.competitorIntent.locationsDescription")}</FieldDescription>
            </Field>
          )}

          {effectiveConfig.competitor_intent_type === "generic" && (
            <Field>
              <FieldLabel htmlFor={`${fieldId}-competitors`}>
                {t("guardrails.contentFilter.competitorIntent.competitorsLabel")}
              </FieldLabel>
              <TagsInput
                id={`${fieldId}-competitors`}
                value={effectiveConfig.competitors ?? []}
                onValueChange={(v) => handleNestedArrayChange("competitors", v)}
                tokenSeparators={[","]}
                placeholder={t("guardrails.contentFilter.competitorIntent.typeToAddPlaceholder")}
              />
              <FieldDescription>
                {t("guardrails.contentFilter.competitorIntent.competitorsDescription")}
              </FieldDescription>
            </Field>
          )}

          <Field>
            <FieldLabel htmlFor={`${fieldId}-competitor-comparison`}>
              {t("guardrails.contentFilter.competitorIntent.competitorComparisonLabel")}
            </FieldLabel>
            <Select
              items={competitorComparisonPolicies}
              value={effectiveConfig.policy?.competitor_comparison ?? "refuse"}
              onValueChange={(v: string | null) => v !== null && handlePolicyChange("competitor_comparison", v)}
            >
              <SelectTrigger id={`${fieldId}-competitor-comparison`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {competitorComparisonPolicies.map((policy) => (
                  <SelectItem key={policy.value} value={policy.value} title={policy.label}>
                    {policy.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field>
            <FieldLabel htmlFor={`${fieldId}-possible-competitor-comparison`}>
              {t("guardrails.contentFilter.competitorIntent.possibleCompetitorComparisonLabel")}
            </FieldLabel>
            <Select
              items={possibleCompetitorComparisonPolicies}
              value={effectiveConfig.policy?.possible_competitor_comparison ?? "reframe"}
              onValueChange={(v: string | null) =>
                v !== null && handlePolicyChange("possible_competitor_comparison", v)
              }
            >
              <SelectTrigger id={`${fieldId}-possible-competitor-comparison`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {possibleCompetitorComparisonPolicies.map((policy) => (
                  <SelectItem key={policy.value} value={policy.value} title={policy.label}>
                    {policy.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field>
            <FieldLabel>{t("guardrails.contentFilter.competitorIntent.thresholdsLabel")}</FieldLabel>
            <div className="flex flex-wrap gap-4">
              {THRESHOLDS.map((threshold) => (
                <Field key={threshold.field} className="w-20">
                  <FieldLabel htmlFor={`${fieldId}-${threshold.field}`}>{t(threshold.labelKey)}</FieldLabel>
                  <ThresholdInput
                    id={`${fieldId}-${threshold.field}`}
                    value={effectiveConfig[threshold.field] ?? threshold.fallback}
                    onValueChange={(v) => handleConfigChange(threshold.field, v ?? threshold.fallback)}
                    min={0}
                    max={1}
                    step={0.05}
                  />
                  <FieldDescription>
                    {t("guardrails.contentFilter.competitorIntent.thresholds.example", { value: threshold.hint })}
                  </FieldDescription>
                </Field>
              ))}
            </div>
            <FieldDescription>
              {t("guardrails.contentFilter.competitorIntent.thresholdsHelp.intro")}
              <ul className="mt-1 mb-0 list-disc pl-5">
                <li>
                  <Trans
                    i18nKey="guardrails.contentFilter.competitorIntent.thresholdsHelp.high"
                    components={{ strong: <strong /> }}
                  />
                </li>
                <li>
                  <Trans
                    i18nKey="guardrails.contentFilter.competitorIntent.thresholdsHelp.medium"
                    components={{ strong: <strong /> }}
                  />
                </li>
                <li>
                  <Trans
                    i18nKey="guardrails.contentFilter.competitorIntent.thresholdsHelp.low"
                    components={{ strong: <strong /> }}
                  />
                </li>
              </ul>
              {t("guardrails.contentFilter.competitorIntent.thresholdsHelp.outro")}
            </FieldDescription>
          </Field>
        </FieldGroup>
      </CardContent>
    </Card>
  );
};

export default CompetitorIntentConfiguration;
