import { Info } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import React from "react";

export const DEFAULT_ESCALATION_KEYWORDS = ["LITELLM ESCALATE"];

interface EscalationKeywordsProps {
  keywords: string[];
  onChange: (keywords: string[]) => void;
}

const EscalationKeywords: React.FC<EscalationKeywordsProps> = ({ keywords, onChange }) => {
  return (
    <div className="w-full max-w-none">
      <div className="flex items-center gap-2 mb-1">
        <h4 className="m-0 text-xl font-semibold text-foreground">Escalation Keywords</h4>
        <SimpleTooltip content="Case-sensitive phrases a user can include in their message to force a bump to the next-higher complexity tier when they aren't happy with results. They can force a stronger model, but not choose which one.">
          <Info className="size-4 text-muted-foreground" />
        </SimpleTooltip>
      </div>
      <span className="mb-2 block text-xs text-muted-foreground">
        Optional: when a user message contains one of these phrases, the request is bumped one tier higher than it would
        otherwise route to. Matching is case-sensitive, so &quot;LITELLM ESCALATE&quot; only fires on the exact, shouted
        form. Leave empty to disable.
      </span>
      <MultiSelect
        options={keywords.map((keyword) => ({ label: keyword, value: keyword }))}
        value={keywords}
        onValueChange={onChange}
        placeholder="e.g., LITELLM ESCALATE"
        emptyText="Type to add a phrase"
        allowCustomValues
        className="w-full"
      />
    </div>
  );
};

export default EscalationKeywords;
