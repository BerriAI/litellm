import React from "react";
import { ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import AdaptiveRoutingConfig from "./AdaptiveRoutingConfig";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import ForecastClassifierConfig from "./ForecastClassifierConfig";
import ContextWindowEscalationConfig from "./ContextWindowEscalationConfig";
import ResponseFormatControls from "./ResponseFormatControls";
import StallEscalationConfig from "./StallEscalationConfig";
import { Restricted, restrictedBy } from "./TierRestrictions";
import EscalationKeywords from "./EscalationKeywords";
import KeywordTierRules, { type KeywordTierRule } from "./KeywordTierRules";
import SemanticKeywordMatching from "./SemanticKeywordMatching";
import CompressionControls from "./CompressionControls";
import PlanModeOverrideControls from "./PlanModeOverrideControls";
import { AffinityControls } from "./AffinityControls";
import { ModalityRoutingControls } from "./ModalityRoutingControls";
import HeuristicKeywordOverrides from "./HeuristicKeywordOverrides";
import HousekeepingRoutingControls from "./HousekeepingRoutingControls";
import ReminderMarkers from "./ReminderMarkers";
import type { AutoRouterCompressionState } from "./buildAutoRouterCompression";
import { activeTierName, type TierRow } from "./tier_rows";

interface ComplexityRouterAdvancedSectionsProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  forecast: boolean;
  modelOptions: { value: string; label: string }[];
  classifierEffortOptionsByModel: Record<string, string[] | null | undefined>;
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  showValidationErrors: boolean;
  defaultModel?: string;
  planModeTierOptions: { value: string; label: string }[];
  keywordTierRules: KeywordTierRule[];
  onKeywordTierRulesChange?: (rules: KeywordTierRule[]) => void;
  semanticMatchingEnabled: boolean;
  onSemanticMatchingEnabledChange?: (enabled: boolean) => void;
  embeddingModel?: string;
  onEmbeddingModelChange: (model: string) => void;
  matchThreshold: number;
  onMatchThresholdChange: (threshold: number) => void;
  escalationKeywords: string[];
  onEscalationKeywordsChange?: (keywords: string[]) => void;
  autoRouterCompression: AutoRouterCompressionState;
  onAutoRouterCompressionChange?: (state: AutoRouterCompressionState) => void;
  modelInfo: ModelGroup[];
  tierRows: TierRow[];
  customTierSet: ComplexityRouterConfigValue["custom_tier_set"];
}

const ComplexityRouterAdvancedSections: React.FC<ComplexityRouterAdvancedSectionsProps> = ({
  value,
  onChange,
  forecast,
  modelOptions,
  classifierEffortOptionsByModel,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  showValidationErrors,
  defaultModel,
  planModeTierOptions,
  keywordTierRules,
  onKeywordTierRulesChange,
  semanticMatchingEnabled,
  onSemanticMatchingEnabledChange,
  embeddingModel,
  onEmbeddingModelChange,
  matchThreshold,
  onMatchThresholdChange,
  escalationKeywords,
  onEscalationKeywordsChange,
  autoRouterCompression,
  onAutoRouterCompressionChange,
  modelInfo,
  tierRows,
  customTierSet,
}) => {
  const sections = [
    ...(forecast
      ? [
          {
            key: "classifier",
            label: <strong className="text-foreground font-semibold">Classifier tuning</strong>,
            children: (
              <ForecastClassifierConfig
                section="advanced"
                value={value}
                onChange={onChange}
                modelOptions={modelOptions}
                effortOptionsByModel={classifierEffortOptionsByModel}
              />
            ),
          },
        ]
      : []),
    ...(!forecast
      ? [
          {
            key: "classifier",
            label: <strong className="text-foreground font-semibold">Classification Method</strong>,
            children: (
              <ClassificationMethodConfig
                advancedOnly
                value={value}
                onChange={onChange}
                modelOptions={modelOptions}
                effortOptionsByModel={classifierEffortOptionsByModel}
                customTechnicalKeywords={customTechnicalKeywords}
                onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
                showValidationErrors={showValidationErrors}
                defaultModel={defaultModel}
              />
            ),
          },
        ]
      : []),
    ...(!forecast
      ? [
          {
            key: "keyword-overrides",
            label: <strong className="text-foreground font-semibold">Heuristic Keyword Overrides</strong>,
            children: <HeuristicKeywordOverrides value={value} onChange={onChange} />,
          },
        ]
      : []),
    {
      key: "adaptive",
      label: <strong className="text-foreground font-semibold">Adaptive Routing</strong>,
      children: (
        <Restricted by={restrictedBy(value, "adaptive")}>
          <AdaptiveRoutingConfig value={value} onChange={onChange} />
        </Restricted>
      ),
    },
    {
      key: "affinity",
      label: <strong className="text-foreground font-semibold">Affinity</strong>,
      children: <AffinityControls value={value} onChange={onChange} />,
    },
    {
      key: "modality",
      label: <strong className="text-foreground font-semibold">Modality Routing</strong>,
      children: <ModalityRoutingControls value={value} onChange={onChange} />,
    },
    {
      key: "plan-mode",
      label: <strong className="text-foreground font-semibold">Plan-Mode Override</strong>,
      children: (
        <PlanModeOverrideControls value={value} onChange={onChange} planModeTierOptions={planModeTierOptions} />
      ),
    },
    {
      key: "housekeeping",
      label: <strong className="text-foreground font-semibold">Housekeeping Routing</strong>,
      children: <HousekeepingRoutingControls value={value} onChange={onChange} />,
    },
    {
      key: "reminder-markers",
      label: <strong className="text-foreground font-semibold">Ignore Custom Tags</strong>,
      children: <ReminderMarkers value={value} onChange={onChange} showValidationErrors={showValidationErrors} />,
    },
    {
      key: "context-window",
      label: <strong className="text-foreground font-semibold">Context Window Escalation</strong>,
      children: <ContextWindowEscalationConfig value={value} onChange={onChange} />,
    },
    {
      key: "stall-escalation",
      label: <strong className="text-foreground font-semibold">Stalled Task Escalation</strong>,
      children: (
        <Restricted by={restrictedBy(value, "stallEscalation")}>
          <StallEscalationConfig value={value} onChange={onChange} />
        </Restricted>
      ),
    },
    {
      key: "response",
      label: <strong className="text-foreground font-semibold">Response Format</strong>,
      children: <ResponseFormatControls value={value} onChange={onChange} />,
    },
    ...(onEscalationKeywordsChange
      ? [
          {
            key: "escalation",
            label: <strong className="text-foreground font-semibold">Escalation Keywords</strong>,
            children: (
              <Restricted by={restrictedBy(value, "escalation")}>
                <EscalationKeywords keywords={escalationKeywords} onChange={onEscalationKeywordsChange} />
              </Restricted>
            ),
          },
        ]
      : []),
    ...(onAutoRouterCompressionChange
      ? [
          {
            key: "compression",
            label: <strong className="text-foreground font-semibold">Compression</strong>,
            children: <CompressionControls value={autoRouterCompression} onChange={onAutoRouterCompressionChange} />,
          },
        ]
      : []),
    ...(onKeywordTierRulesChange || onSemanticMatchingEnabledChange
      ? [
          {
            key: "keyword-semantic",
            label: <strong className="text-foreground font-semibold">Keyword/Semantic Matching</strong>,
            children: (
              <>
                {onKeywordTierRulesChange && (
                  <KeywordTierRules
                    rules={keywordTierRules}
                    onChange={onKeywordTierRulesChange}
                    tierLabels={value.tier_labels}
                    tierNames={customTierSet || forecast ? tierRows.map(activeTierName).filter(Boolean) : undefined}
                  />
                )}
                {onKeywordTierRulesChange && onSemanticMatchingEnabledChange && <Separator className="my-4" />}
                {onSemanticMatchingEnabledChange && (
                  <SemanticKeywordMatching
                    enabled={semanticMatchingEnabled}
                    onEnabledChange={onSemanticMatchingEnabledChange}
                    embeddingModel={embeddingModel}
                    onEmbeddingModelChange={onEmbeddingModelChange}
                    matchThreshold={matchThreshold}
                    onMatchThresholdChange={onMatchThresholdChange}
                    modelInfo={modelInfo}
                    showValidationErrors={showValidationErrors}
                  />
                )}
              </>
            ),
          },
        ]
      : []),
  ];

  const groups = [
    { label: "Classifier tuning", keys: ["classifier", "keyword-overrides", "reminder-markers"] },
    {
      label: "Routing rules and recovery",
      keys: [
        "modality",
        "plan-mode",
        "housekeeping",
        "context-window",
        "stall-escalation",
        "escalation",
        "keyword-semantic",
      ],
    },
    { label: "Sessions and efficiency", keys: ["affinity", "adaptive", "compression"] },
    { label: "Compatibility", keys: ["response"] },
  ];
  const [openGroups, setOpenGroups] = React.useState<string[]>(() =>
    showValidationErrors ? groups.map((group) => group.label) : [],
  );
  const [previousValidation, setPreviousValidation] = React.useState(showValidationErrors);
  if (previousValidation !== showValidationErrors) {
    setPreviousValidation(showValidationErrors);
    if (showValidationErrors) setOpenGroups(groups.map((group) => group.label));
  }
  return (
    <div>
      {groups.map((group) => (
        <Collapsible
          key={group.label}
          open={openGroups.includes(group.label)}
          onOpenChange={(open) =>
            setOpenGroups((current) =>
              open ? [...current, group.label] : current.filter((label) => label !== group.label),
            )
          }
          className="border-b border-border last:border-b-0"
        >
          <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-3 text-left font-medium">
            <ChevronRight className="size-4 shrink-0 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
            {group.label}
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-6 px-4 pb-4">
            {sections
              .filter(
                ({ key }) =>
                  group.keys.includes(key) &&
                  (!forecast || !["adaptive", "context-window", "escalation"].includes(key)),
              )
              .map(({ key, label, children }) => (
                <section key={key} className="space-y-3">
                  {key !== "classifier" && <h4>{label}</h4>}
                  {children}
                </section>
              ))}
          </CollapsibleContent>
        </Collapsible>
      ))}
    </div>
  );
};

export default ComplexityRouterAdvancedSections;
