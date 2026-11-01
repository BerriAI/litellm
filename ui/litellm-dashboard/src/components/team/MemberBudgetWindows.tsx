import { useId } from "react";
import { CalendarRange } from "lucide-react";
import { BudgetWindowsEditor, type BudgetWindowEntry } from "../key_team_helpers/BudgetWindowsEditor";
import { getBudgetDurationLabel } from "../common_components/budget_duration_dropdown";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

interface MemberBudgetWindowsProps {
  value: BudgetWindowEntry[];
  onChange: (windows: BudgetWindowEntry[]) => void;
  defaultWindows?: BudgetWindowEntry[];
}

export function MemberBudgetWindows({ value, onChange, defaultWindows }: MemberBudgetWindowsProps) {
  const id = useId();
  const enabled = value.length > 0;
  const isMember = defaultWindows !== undefined;
  const inherits = isMember && !enabled && defaultWindows.length > 0;
  const label = isMember ? "Custom budget windows" : "Enable budget windows";

  const description = enabled
    ? "Apply multiple spending caps at the same time."
    : "Optional. Leave off to keep your existing budget settings.";

  const toggleWindows = (checked: boolean) => {
    if (!checked) {
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
    <div className="space-y-4 rounded-lg border border-border bg-muted/20 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <CalendarRange className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden="true" />
          <div className="space-y-1">
            <Label htmlFor={id}>{label}</Label>
            <p className="text-xs text-muted-foreground">
              {inherits ? "Off uses the team defaults below." : description}
            </p>
          </div>
        </div>
        <Switch id={id} checked={enabled} onCheckedChange={toggleWindows} />
      </div>

      {!enabled && !inherits && (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Add a daily cap to stop a runaway session and a monthly cap to control total spend for each member in this
            team.
          </p>
          <div
            role="img"
            aria-label="Example: a 20 dollar daily cap and a 200 dollar monthly cap both apply"
            className="flex items-center gap-2 text-sm"
          >
            <div className="min-w-0 flex-1 rounded-md border border-border bg-background p-2.5">
              <div className="text-xs text-muted-foreground">Daily cap</div>
              <div className="mt-1 font-medium">
                $20 <span className="text-xs font-normal text-muted-foreground">/ day</span>
              </div>
              <div className="mt-2 h-1 rounded-full bg-info/15">
                <div className="h-1 w-1/3 rounded-full bg-info" />
              </div>
            </div>
            <span aria-hidden="true" className="text-muted-foreground">
              +
            </span>
            <div className="min-w-0 flex-1 rounded-md border border-border bg-background p-2.5">
              <div className="text-xs text-muted-foreground">Monthly cap</div>
              <div className="mt-1 font-medium">
                $200 <span className="text-xs font-normal text-muted-foreground">/ month</span>
              </div>
              <div className="mt-2 h-1 rounded-full bg-info/15">
                <div className="h-1 w-2/3 rounded-full bg-info" />
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Example only. Both caps apply together. Each resets on its own schedule.
          </p>
        </div>
      )}

      {inherits && (
        <div className="rounded-md bg-background p-3 text-sm">
          <p className="font-medium">Using team defaults</p>
          <p className="mt-1 text-muted-foreground">
            {defaultWindows
              .map((window) => `$${window.max_budget} ${getBudgetDurationLabel(window.budget_duration)}`)
              .join(", ")}
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            Turn on custom windows to replace these limits for this member.
          </p>
        </div>
      )}

      {enabled && (
        <div className="space-y-3">
          <BudgetWindowsEditor value={value} onChange={onChange} />
          <p className="text-xs text-muted-foreground">
            Spending pauses when any cap is reached. Each window resets independently. Existing single-budget limits
            also apply.
          </p>
        </div>
      )}
    </div>
  );
}
