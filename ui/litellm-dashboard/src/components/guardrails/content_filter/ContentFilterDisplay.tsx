import React from "react";
import { Card, Text, Badge } from "@tremor/react";
import PatternTable from "./PatternTable";
import KeywordTable from "./KeywordTable";

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
  enabled: boolean;
  action: "BLOCK" | "MASK";
  severity_threshold?: string;
}

interface ContentFilterDisplayProps {
  patterns: Pattern[];
  blockedWords: BlockedWord[];
  contentCategories?: ContentCategory[];
  readOnly?: boolean;
  onPatternActionChange?: (id: string, action: "BLOCK" | "MASK") => void;
  onPatternRemove?: (id: string) => void;
  onBlockedWordUpdate?: (id: string, field: string, value: any) => void;
  onBlockedWordRemove?: (id: string) => void;
  onCategoryUpdate?: (id: string, field: string, value: any) => void;
  onCategoryRemove?: (id: string) => void;
}

const ContentFilterDisplay: React.FC<ContentFilterDisplayProps> = ({
  patterns,
  blockedWords,
  contentCategories = [],
  readOnly = true,
  onPatternActionChange,
  onPatternRemove,
  onBlockedWordUpdate,
  onBlockedWordRemove,
  onCategoryUpdate,
  onCategoryRemove,
}) => {
  if (patterns.length === 0 && blockedWords.length === 0 && contentCategories.length === 0) {
    return null;
  }

  // No-op handlers for read-only mode
  const noOp = () => {};

  return (
    <>
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
      {contentCategories.length > 0 && (
        <Card className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <Text className="text-lg font-semibold">Content Categories</Text>
            <Badge color="indigo">{contentCategories.length} categories configured</Badge>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Category
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Action
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Severity Threshold
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {contentCategories.map((category) => (
                  <tr key={category.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {category.category}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <Badge color={category.action === "BLOCK" ? "red" : "yellow"}>
                        {category.action}
                      </Badge>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {category.severity_threshold || "medium"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <Badge color={category.enabled ? "green" : "gray"}>
                        {category.enabled ? "Enabled" : "Disabled"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}    </>
  );
};

export default ContentFilterDisplay;

