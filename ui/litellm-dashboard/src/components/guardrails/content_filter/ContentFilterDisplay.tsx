import React from "react";
import { Card, Text, Badge } from "@tremor/react";
import PatternTable from "./PatternTable";
import KeywordTable from "./KeywordTable";
import CategoryTable from "./CategoryTable";

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

interface ContentCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

interface ContentFilterDisplayProps {
  patterns: Pattern[];
  blockedWords: BlockedWord[];
  categories?: ContentCategory[];
  readOnly?: boolean;
  onPatternActionChange?: (id: string, action: "BLOCK" | "MASK") => void;
  onPatternRemove?: (id: string) => void;
  onBlockedWordUpdate?: (id: string, field: string, value: any) => void;
  onBlockedWordRemove?: (id: string) => void;
  onCategoryActionChange?: (id: string, action: "BLOCK" | "MASK") => void;
  onCategorySeverityChange?: (id: string, severity: "high" | "medium" | "low") => void;
  onCategoryRemove?: (id: string) => void;
}

const ContentFilterDisplay: React.FC<ContentFilterDisplayProps> = ({
  patterns,
  blockedWords,
  categories = [],
  readOnly = true,
  onPatternActionChange,
  onPatternRemove,
  onBlockedWordUpdate,
  onBlockedWordRemove,
  onCategoryActionChange,
  onCategorySeverityChange,
  onCategoryRemove,
}) => {
  if (patterns.length === 0 && blockedWords.length === 0 && categories.length === 0) {
    return null;
  }

  // No-op handlers for read-only mode
  const noOp = () => {};

  return (
    <>
      {categories.length > 0 && (
        <Card className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <Text className="text-lg font-semibold">Content Categories</Text>
            <Badge color="blue">{categories.length} categories configured</Badge>
          </div>
          <CategoryTable
            categories={categories}
            onActionChange={readOnly ? undefined : onCategoryActionChange}
            onSeverityChange={readOnly ? undefined : onCategorySeverityChange}
            onRemove={readOnly ? undefined : onCategoryRemove}
            readOnly={readOnly}
          />
        </Card>
      )}

      {patterns.length > 0 && (
        <Card className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <Text className="text-lg font-semibold">Pattern Detection</Text>
            <Badge color="blue">{patterns.length} patterns configured</Badge>
          </div>
          <PatternTable
            patterns={patterns}
            onActionChange={readOnly ? noOp : (onPatternActionChange || noOp)}
            onRemove={readOnly ? noOp : (onPatternRemove || noOp)}
          />
        </Card>
      )}

      {blockedWords.length > 0 && (
        <Card className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <Text className="text-lg font-semibold">Blocked Keywords</Text>
            <Badge color="blue">{blockedWords.length} keywords configured</Badge>
          </div>
          <KeywordTable
            keywords={blockedWords}
            onActionChange={readOnly ? noOp : (onBlockedWordUpdate || noOp)}
            onRemove={readOnly ? noOp : (onBlockedWordRemove || noOp)}
          />
        </Card>
      )}
    </>
  );
};

export default ContentFilterDisplay;

