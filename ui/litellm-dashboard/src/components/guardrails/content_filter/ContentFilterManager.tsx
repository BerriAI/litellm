import { Alert, Divider, Typography } from "antd";
import React, { useEffect, useState } from "react";
import ContentFilterConfiguration from "./ContentFilterConfiguration";
import ContentFilterDisplay from "./ContentFilterDisplay";
import type { CompetitorIntentConfig } from "./CompetitorIntentConfiguration";

const { Text } = Typography

interface Pattern {
  id: string;
  type: "prebuilt" | "custom";
  name: string;
  display_name?: string;
  pattern?: string;
  action: "BLOCK" | "MASK";
}

interface BlockedWord {
  id: string;
  keyword: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

interface SelectedContentCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

interface ContentCategory {
  name: string;
  display_name: string;
  description: string;
  default_action: string;
}

interface GuardrailSettings {
  content_filter_settings?: {
    prebuilt_patterns: Array<{
      name: string;
      display_name: string;
      category: string;
      description: string;
    }>;
    pattern_categories: string[];
    supported_actions: string[];
    content_categories?: ContentCategory[];
  };
}

interface ContentFilterManagerProps {
  guardrailData: any;
  guardrailSettings: GuardrailSettings | null;
  isEditing: boolean;
  accessToken: string | null;
  onDataChange?: (
    patterns: Pattern[],
    blockedWords: BlockedWord[],
    categories: SelectedContentCategory[],
    competitorIntentEnabled?: boolean,
    competitorIntentConfig?: CompetitorIntentConfig | null
  ) => void;
  onUnsavedChanges?: (hasChanges: boolean) => void;
}

const ContentFilterManager: React.FC<ContentFilterManagerProps> = ({
  guardrailData,
  guardrailSettings,
  isEditing,
  accessToken,
  onDataChange,
  onUnsavedChanges,
}) => {
  const [selectedPatterns, setSelectedPatterns] = useState<Pattern[]>([]);
  const [blockedWords, setBlockedWords] = useState<BlockedWord[]>([]);
  const [selectedContentCategories, setSelectedContentCategories] = useState<SelectedContentCategory[]>([]);
  const [originalPatterns, setOriginalPatterns] = useState<Pattern[]>([]);
  const [originalBlockedWords, setOriginalBlockedWords] = useState<BlockedWord[]>([]);
  const [originalContentCategories, setOriginalContentCategories] = useState<SelectedContentCategory[]>([]);
  const [competitorIntentEnabled, setCompetitorIntentEnabled] = useState(false);
  const [competitorIntentConfig, setCompetitorIntentConfig] = useState<CompetitorIntentConfig | null>(null);
  const [originalCompetitorIntentEnabled, setOriginalCompetitorIntentEnabled] = useState(false);
  const [originalCompetitorIntentConfig, setOriginalCompetitorIntentConfig] = useState<CompetitorIntentConfig | null>(null);

  // Load data from guardrail on mount or when guardrailData changes
  useEffect(() => {
    if (guardrailData?.litellm_params?.patterns) {
      const patterns = guardrailData.litellm_params.patterns.map((p: any, index: number) => ({
        id: `pattern-${index}`,
        type: p.pattern_type === "prebuilt" ? "prebuilt" : "custom",
        name: p.pattern_name || p.name,
        display_name: p.display_name,
        pattern: p.pattern,
        action: p.action || "BLOCK",
      }));
      setSelectedPatterns(patterns);
      setOriginalPatterns(patterns);
    } else {
      setSelectedPatterns([]);
      setOriginalPatterns([]);
    }

    if (guardrailData?.litellm_params?.blocked_words) {
      const words = guardrailData.litellm_params.blocked_words.map((w: any, index: number) => ({
        id: `word-${index}`,
        keyword: w.keyword,
        action: w.action || "BLOCK",
        description: w.description,
      }));
      setBlockedWords(words);
      setOriginalBlockedWords(words);
    } else {
      setBlockedWords([]);
      setOriginalBlockedWords([]);
    }

    if (guardrailData?.litellm_params?.categories?.length > 0) {
      const contentCategoriesMap = guardrailSettings?.content_filter_settings?.content_categories
        ? Object.fromEntries(
          guardrailSettings.content_filter_settings.content_categories.map((c) => [c.name, c])
        )
        : {};
      const categories = guardrailData.litellm_params.categories.map((c: any, index: number) => {
        const meta = contentCategoriesMap[c.category];
        return {
          id: `category-${index}`,
          category: c.category,
          display_name: meta?.display_name ?? c.category,
          action: (c.action || "BLOCK") as "BLOCK" | "MASK",
          severity_threshold: (c.severity_threshold || "medium") as "high" | "medium" | "low",
        };
      });
      setSelectedContentCategories(categories);
      setOriginalContentCategories(categories);
    } else {
      setSelectedContentCategories([]);
      setOriginalContentCategories([]);
    }

    const cic = guardrailData?.litellm_params?.competitor_intent_config;
    if (cic && typeof cic === "object") {
      const enabled = !!(cic.brand_self && Array.isArray(cic.brand_self) && cic.brand_self.length > 0);
      const config: CompetitorIntentConfig = {
        competitor_intent_type: cic.competitor_intent_type ?? "airline",
        brand_self: Array.isArray(cic.brand_self) ? cic.brand_self : [],
        locations: Array.isArray(cic.locations) ? cic.locations : [],
        competitors: Array.isArray(cic.competitors) ? cic.competitors : [],
        policy: cic.policy ?? { competitor_comparison: "refuse", possible_competitor_comparison: "reframe" },
        threshold_high: typeof cic.threshold_high === "number" ? cic.threshold_high : 0.7,
        threshold_medium: typeof cic.threshold_medium === "number" ? cic.threshold_medium : 0.45,
        threshold_low: typeof cic.threshold_low === "number" ? cic.threshold_low : 0.3,
      };
      setCompetitorIntentEnabled(enabled);
      setCompetitorIntentConfig(config);
      setOriginalCompetitorIntentEnabled(enabled);
      setOriginalCompetitorIntentConfig(config);
    } else {
      setCompetitorIntentEnabled(false);
      setCompetitorIntentConfig(null);
      setOriginalCompetitorIntentEnabled(false);
      setOriginalCompetitorIntentConfig(null);
    }
  }, [guardrailData, guardrailSettings?.content_filter_settings?.content_categories]);

  // Notify parent component when data changes
  useEffect(() => {
    if (onDataChange) {
      onDataChange(
        selectedPatterns,
        blockedWords,
        selectedContentCategories,
        competitorIntentEnabled,
        competitorIntentConfig
      );
    }
  }, [
    selectedPatterns,
    blockedWords,
    selectedContentCategories,
    competitorIntentEnabled,
    competitorIntentConfig,
    onDataChange,
  ]);

  // Detect unsaved changes
  const hasUnsavedChanges = React.useMemo(() => {
    const hasPatternChanges = JSON.stringify(selectedPatterns) !== JSON.stringify(originalPatterns);
    const hasWordChanges = JSON.stringify(blockedWords) !== JSON.stringify(originalBlockedWords);
    const hasCategoryChanges = JSON.stringify(selectedContentCategories) !== JSON.stringify(originalContentCategories);
    const hasCompetitorIntentChanges =
      competitorIntentEnabled !== originalCompetitorIntentEnabled ||
      JSON.stringify(competitorIntentConfig) !== JSON.stringify(originalCompetitorIntentConfig);
    return hasPatternChanges || hasWordChanges || hasCategoryChanges || hasCompetitorIntentChanges;
  }, [
    selectedPatterns,
    blockedWords,
    selectedContentCategories,
    competitorIntentEnabled,
    competitorIntentConfig,
    originalPatterns,
    originalBlockedWords,
    originalContentCategories,
    originalCompetitorIntentEnabled,
    originalCompetitorIntentConfig,
  ]);

  useEffect(() => {
    if (isEditing && onUnsavedChanges) {
      onUnsavedChanges(hasUnsavedChanges);
    }
  }, [hasUnsavedChanges, isEditing, onUnsavedChanges]);

  // Check if this is a content filter guardrail
  if (guardrailData?.litellm_params?.guardrail !== "litellm_content_filter") {
    return null;
  }

  // Read-only display mode
  if (!isEditing) {
    return (
      <ContentFilterDisplay
        patterns={selectedPatterns}
        blockedWords={blockedWords}
        categories={selectedContentCategories}
        readOnly={true}
      />
    );
  }

  // Edit mode
  return (
    <>
      <Divider orientation="left">Content Filter Configuration</Divider>
      {hasUnsavedChanges && (
        <Alert
          type="warning"
          showIcon
          className="mb-4"
          message={
            <Text>
              You have unsaved changes to patterns or keywords. Remember to click &quot;Save Changes&quot; at the
              bottom.
            </Text>
          }
        />
      )}
      <div className="mb-6">
        {guardrailSettings && guardrailSettings.content_filter_settings && (
          <ContentFilterConfiguration
            prebuiltPatterns={guardrailSettings.content_filter_settings.prebuilt_patterns || []}
            categories={guardrailSettings.content_filter_settings.pattern_categories || []}
            selectedPatterns={selectedPatterns}
            blockedWords={blockedWords}
            onPatternAdd={(pattern) => setSelectedPatterns([...selectedPatterns, pattern])}
            onPatternRemove={(id) => setSelectedPatterns(selectedPatterns.filter((p) => p.id !== id))}
            onPatternActionChange={(id, action) =>
              setSelectedPatterns(selectedPatterns.map((p) => (p.id === id ? { ...p, action } : p)))
            }
            onBlockedWordAdd={(word) => setBlockedWords([...blockedWords, word])}
            onBlockedWordRemove={(id) => setBlockedWords(blockedWords.filter((w) => w.id !== id))}
            onBlockedWordUpdate={(id, field, value) =>
              setBlockedWords(blockedWords.map((w) => (w.id === id ? { ...w, [field]: value } : w)))
            }
            onFileUpload={(content: string) => {
              console.log("File uploaded:", content);
            }}
            accessToken={accessToken}
            contentCategories={guardrailSettings.content_filter_settings.content_categories || []}
            selectedContentCategories={selectedContentCategories}
            onContentCategoryAdd={(category) =>
              setSelectedContentCategories([...selectedContentCategories, category])
            }
            onContentCategoryRemove={(id) =>
              setSelectedContentCategories(selectedContentCategories.filter((c) => c.id !== id))
            }
            onContentCategoryUpdate={(id, field, value) =>
              setSelectedContentCategories(
                selectedContentCategories.map((c) => (c.id === id ? { ...c, [field]: value } : c))
              )
            }
            competitorIntentEnabled={competitorIntentEnabled}
            competitorIntentConfig={competitorIntentConfig}
            onCompetitorIntentChange={(enabled, config) => {
              setCompetitorIntentEnabled(enabled);
              setCompetitorIntentConfig(config);
            }}
          />
        )}
      </div>
    </>
  );
};

export default ContentFilterManager;

// Helper function to format data for API
export const formatContentFilterDataForAPI = (
  patterns: Pattern[],
  blockedWords: BlockedWord[],
  categories?: SelectedContentCategory[],
  competitorIntentEnabled?: boolean,
  competitorIntentConfig?: CompetitorIntentConfig | null
) => {
  const result: {
    patterns: any[];
    blocked_words: any[];
    categories?: any[];
    competitor_intent_config?: any;
  } = {
    patterns: patterns.map((p) => ({
      pattern_type: p.type === "prebuilt" ? "prebuilt" : "regex",
      pattern_name: p.type === "prebuilt" ? p.name : undefined,
      pattern: p.type === "custom" ? p.pattern : undefined,
      name: p.name,
      action: p.action,
    })),
    blocked_words: blockedWords.map((w) => ({
      keyword: w.keyword,
      action: w.action,
      description: w.description,
    })),
  };
  if (categories !== undefined) {
    result.categories = categories.map((c) => ({
      category: c.category,
      enabled: true,
      action: c.action,
      severity_threshold: c.severity_threshold || "medium",
    }));
  }
  if (competitorIntentEnabled && competitorIntentConfig && competitorIntentConfig.brand_self.length > 0) {
    result.competitor_intent_config = {
      competitor_intent_type: competitorIntentConfig.competitor_intent_type,
      brand_self: competitorIntentConfig.brand_self,
      locations: competitorIntentConfig.locations?.length
        ? competitorIntentConfig.locations
        : undefined,
      competitors:
        competitorIntentConfig.competitor_intent_type === "generic" &&
        competitorIntentConfig.competitors?.length
          ? competitorIntentConfig.competitors
          : undefined,
      policy: competitorIntentConfig.policy,
      threshold_high: competitorIntentConfig.threshold_high,
      threshold_medium: competitorIntentConfig.threshold_medium,
      threshold_low: competitorIntentConfig.threshold_low,
    };
  }
  return result;
};
