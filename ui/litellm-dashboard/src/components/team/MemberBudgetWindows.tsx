import { useId } from "react";
import { BudgetWindowsEditor, type BudgetWindowEntry } from "../key_team_helpers/BudgetWindowsEditor";
import { getBudgetDurationLabel } from "../common_components/budget_duration_dropdown";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

interface MemberBudgetWindowsProps {
  value: BudgetWindowEntry[];
  onChange: (windows: BudgetWindowEntry[]) => void;
  defaultWindows?: BudgetWindowEntry[];
}

export function MemberBudgetWindows({ value, onChange, defaultWindows }: MemberBudgetWindowsProps) {
  const id = useId();
  const inherits = defaultWindows !== undefined && value.length === 0;
  const selectMode = (selection: unknown) => {
    if (selection === "default") {
      onChange([]);
      return;
    }
    if (defaultWindows?.length) {
      onChange(defaultWindows.map((window) => ({ ...window })));
      return;
    }
    onChange([{ budget_duration: "24h", max_budget: null }]);
  };

  return (
    <div className="space-y-3">
      {defaultWindows !== undefined && (
        <RadioGroup
          aria-label="Member budget windows"
          value={inherits ? "default" : "custom"}
          onValueChange={selectMode}
        >
          <div className="flex items-center gap-2">
            <RadioGroupItem id={`${id}-default`} value="default" />
            <Label htmlFor={`${id}-default`}>Use team defaults</Label>
          </div>
          <div className="flex items-center gap-2">
            <RadioGroupItem id={`${id}-custom`} value="custom" />
            <Label htmlFor={`${id}-custom`}>Custom budget windows</Label>
          </div>
        </RadioGroup>
      )}
      {inherits ? (
        <p className="text-sm text-muted-foreground">
          {defaultWindows.length
            ? defaultWindows
                .map((window) => `$${window.max_budget} ${getBudgetDurationLabel(window.budget_duration)}`)
                .join(", ")
            : "No default budget windows set"}
        </p>
      ) : (
        <BudgetWindowsEditor value={value} onChange={onChange} />
      )}
      <p className="text-sm text-muted-foreground">
        {defaultWindows === undefined
          ? "Each window limits one member's spend in this team. All windows apply together."
          : "Custom windows replace the team defaults for this member. Existing single-budget limits also apply."}
      </p>
    </div>
  );
}
