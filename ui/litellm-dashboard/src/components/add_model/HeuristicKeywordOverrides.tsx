import React from "react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const fields = [
  ["code_keywords", "Code keywords"],
  ["reasoning_keywords", "Reasoning keywords"],
  ["technical_keywords", "Technical keywords"],
  ["simple_keywords", "Simple keywords"],
] as const;

const HeuristicKeywordOverrides: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => (
  <div className="space-y-3">
    <p className="text-sm text-muted-foreground">
      Each list replaces the built-in keyword list of the same name for the heuristic scorer. Leave a list empty to keep
      the built-in one. To add technical terms without replacing the list, use custom technical keywords under
      Classification Method.
    </p>
    {fields.map(([key, label]) => {
      const keywords = value[key] ?? [];
      return (
        <div key={key}>
          <strong className="mb-1 block font-semibold">{label}</strong>
          <MultiSelect
            options={keywords.map((keyword) => ({ label: keyword, value: keyword }))}
            value={keywords}
            onValueChange={(next) => onChange({ ...value, [key]: next.length > 0 ? next : undefined })}
            placeholder={`Add ${label.toLowerCase()}`}
            emptyText="Type to add a keyword"
            allowCustomValues
            className="w-full"
          />
        </div>
      );
    })}
  </div>
);

export default HeuristicKeywordOverrides;
