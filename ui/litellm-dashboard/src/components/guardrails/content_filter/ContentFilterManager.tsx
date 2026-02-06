import React, { useState, useEffect } from "react";
import { Divider } from "antd";
import ContentFilterConfiguration from "./ContentFilterConfiguration";
import ContentFilterDisplay from "./ContentFilterDisplay";

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
  };
}

interface ContentFilterManagerProps {
  guardrailData: any;
  guardrailSettings: GuardrailSettings | null;
  isEditing: boolean;
  accessToken: string | null;
  onDataChange?: (patterns: Pattern[], blockedWords: BlockedWord[]) => void;
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
  const [originalPatterns, setOriginalPatterns] = useState<Pattern[]>([]);
  const [originalBlockedWords, setOriginalBlockedWords] = useState<BlockedWord[]>([]);

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
  }, [guardrailData]);

  // Notify parent component when data changes
  useEffect(() => {
    if (onDataChange) {
      onDataChange(selectedPatterns, blockedWords);
    }
  }, [selectedPatterns, blockedWords, onDataChange]);

  // Detect unsaved changes
  const hasUnsavedChanges = React.useMemo(() => {
    const hasPatternChanges = JSON.stringify(selectedPatterns) !== JSON.stringify(originalPatterns);
    const hasWordChanges = JSON.stringify(blockedWords) !== JSON.stringify(originalBlockedWords);
    return hasPatternChanges || hasWordChanges;
  }, [selectedPatterns, blockedWords, originalPatterns, originalBlockedWords]);

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
    return <ContentFilterDisplay patterns={selectedPatterns} blockedWords={blockedWords} readOnly={true} />;
  }

  // Edit mode
  return (
    <>
      <Divider orientation="left">Content Filter Configuration</Divider>
      {hasUnsavedChanges && (
        <div className="mb-4 px-4 py-3 bg-yellow-50 border border-yellow-200 rounded-md">
          <p className="text-sm text-yellow-800 font-medium">
            ⚠️ You have unsaved changes to patterns or keywords. Remember to click &quot;Save Changes&quot; at the
            bottom.
          </p>
        </div>
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
          />
        )}
      </div>
    </>
  );
};

export default ContentFilterManager;

// Helper function to format data for API
export const formatContentFilterDataForAPI = (patterns: Pattern[], blockedWords: BlockedWord[]) => {
  return {
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
};
