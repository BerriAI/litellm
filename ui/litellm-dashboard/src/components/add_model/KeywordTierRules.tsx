import { DeleteOutlined, InfoCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, Card, Empty, Select as AntdSelect, Tooltip, Typography } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";

import { emptyKeywordTierRuleIndexes } from "./complexity_router_keywords";

const { Text } = Typography;

export type ComplexityTier = "SIMPLE" | "MEDIUM" | "COMPLEX" | "REASONING";

export interface KeywordTierRule {
  id: string;
  keywords: string[];
  tier: ComplexityTier;
}

interface KeywordTierRulesProps {
  rules: KeywordTierRule[];
  onChange: (rules: KeywordTierRule[]) => void;
  tierLabels?: Partial<Record<ComplexityTier, string>>;
}

const DEFAULT_TIER_LABELS: Record<ComplexityTier, string> = {
  SIMPLE: "Simple",
  MEDIUM: "Medium",
  COMPLEX: "Complex",
  REASONING: "Reasoning",
};

const TIER_ORDER: ComplexityTier[] = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"];

export const tierOptions = (
  tierLabels: Partial<Record<ComplexityTier, string>> | undefined,
): { value: ComplexityTier; label: string }[] =>
  TIER_ORDER.map((tier) => ({ value: tier, label: tierLabels?.[tier]?.trim() || DEFAULT_TIER_LABELS[tier] }));

// A row exists only because the caller asked for it, so it reports its own gap straight away
// rather than waiting for a submit; the submit button is disabled while one is outstanding, so
// there is no failed attempt left to surface it.
const KeywordTierRules: React.FC<KeywordTierRulesProps> = ({ rules, onChange, tierLabels }) => {
  const { t } = useTranslation("gateway");
  const emptyRuleIndexes = new Set(emptyKeywordTierRuleIndexes(rules));
  const [drafts, setDrafts] = React.useState<Record<string, string>>({});

  const setDraft = (id: string, text: string) => setDrafts((current) => ({ ...current, [id]: text }));

  // The dropdown is kept closed, which leaves antd nothing for Enter to select, so a typed keyword
  // would only become a tag on blur. Submitting used to supply that blur; the button is disabled
  // while the row reads as empty, so Enter has to commit the word itself or the row cannot be filled.
  const commitDraft = (rule: KeywordTierRule) => {
    const keyword = (drafts[rule.id] ?? "").trim();
    if (!keyword) return;
    updateRule(rule.id, { keywords: [...rule.keywords, keyword] });
    setDraft(rule.id, "");
  };

  const commitDraftOnEnter = (rule: KeywordTierRule) => (event: React.KeyboardEvent<HTMLElement>) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    commitDraft(rule);
  };

  const replaceKeywords = (rule: KeywordTierRule) => (keywords: string[]) => {
    updateRule(rule.id, { keywords });
    setDraft(rule.id, "");
  };

  const addRule = () => {
    onChange([...rules, { id: `${Date.now()}`, keywords: [], tier: "COMPLEX" }]);
  };

  const updateRule = (id: string, updates: Partial<Omit<KeywordTierRule, "id">>) => {
    onChange(rules.map((rule) => (rule.id === id ? { ...rule, ...updates } : rule)));
  };

  const removeRule = (id: string) => {
    onChange(rules.filter((rule) => rule.id !== id));
  };

  return (
    <div className="w-full max-w-none">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Typography.Title level={4} style={{ margin: 0 }}>
            {t("models.autoRouters.details.keywordRules.title")}
          </Typography.Title>
          <Tooltip title={t("models.autoRouters.details.keywordRules.tooltip")}>
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>
        <Button icon={<PlusOutlined />} onClick={addRule}>
          {t("models.autoRouters.details.keywordRules.add")}
        </Button>
      </div>
      <Text type="secondary" style={{ display: "block", marginBottom: 16 }}>
        {t("models.autoRouters.details.keywordRules.description")}
      </Text>

      {rules.length === 0 ? (
        <Card className="bg-gray-50">
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={t("models.autoRouters.details.keywordRules.empty")}
          />
        </Card>
      ) : (
        <div className="flex flex-col gap-3">
          {rules.map((rule, index) => (
            <Card key={rule.id} size="small">
              <div className="flex items-end gap-3">
                <div className="flex-1">
                  <Text strong style={{ display: "block", marginBottom: 8 }}>
                    {t("models.autoRouters.details.keywordRules.keywords", { number: index + 1 })}
                  </Text>
                  <AntdSelect
                    mode="tags"
                    value={rule.keywords}
                    onChange={replaceKeywords(rule)}
                    searchValue={drafts[rule.id] ?? ""}
                    onSearch={(text) => setDraft(rule.id, text)}
                    onInputKeyDown={commitDraftOnEnter(rule)}
                    onBlur={() => commitDraft(rule)}
                    placeholder={t("models.autoRouters.details.keywordRules.placeholder")}
                    tokenSeparators={[","]}
                    open={false}
                    suffixIcon={null}
                    style={{ width: "100%" }}
                    allowClear
                    status={emptyRuleIndexes.has(index) ? "error" : undefined}
                  />
                  {emptyRuleIndexes.has(index) && (
                    <Text type="danger" style={{ fontSize: 12 }}>
                      {t("models.autoRouters.details.keywordRules.required")}
                    </Text>
                  )}
                </div>
                <div style={{ width: 220 }}>
                  <Text strong style={{ display: "block", marginBottom: 8 }}>
                    {t("models.autoRouters.details.keywordRules.routeTo")}
                  </Text>
                  <AntdSelect
                    value={rule.tier}
                    onChange={(tier: ComplexityTier) => updateRule(rule.id, { tier })}
                    options={tierOptions(tierLabels).map((option) => ({
                      ...option,
                      label:
                        tierLabels?.[option.value]?.trim() ||
                        t(`models.autoRouters.details.tiers.${option.value.toLowerCase()}.label`, {
                          defaultValue: option.label,
                        }),
                    }))}
                    style={{ width: "100%" }}
                  />
                </div>
                <Button
                  danger
                  type="text"
                  icon={<DeleteOutlined />}
                  aria-label={t("models.autoRouters.details.keywordRules.remove", { number: index + 1 })}
                  onClick={() => removeRule(rule.id)}
                />
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default KeywordTierRules;
