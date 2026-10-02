import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { MemberBudgetUpdateMode } from "./memberBudgetReset";

const OPTIONS: Record<MemberBudgetUpdateMode, string> = {
  keep: "Keep custom budgets",
  raise: "Raise smaller budgets",
  lower: "Lower larger budgets",
  both: "Raise and lower",
};

interface MemberBudgetUpdateSelectProps {
  id: string;
  value: MemberBudgetUpdateMode;
  onChange: (value: MemberBudgetUpdateMode) => void;
  disabled?: boolean;
}

export default function MemberBudgetUpdateSelect({ id, value, onChange, disabled }: MemberBudgetUpdateSelectProps) {
  return (
    <div className="space-y-2">
      <Select<MemberBudgetUpdateMode>
        items={OPTIONS}
        value={value}
        onValueChange={(next) => {
          if (next !== null) onChange(next);
        }}
        disabled={disabled}
      >
        <SelectTrigger
          id={id}
          aria-describedby={`${id}-description`}
          className="min-h-9 w-full whitespace-normal data-[size=default]:h-auto *:data-[slot=select-value]:line-clamp-none"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent className="min-w-56">
          {Object.entries(OPTIONS).map(([mode, label]) => (
            <SelectItem key={mode} value={mode}>
              {label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p id={`${id}-description`} className="text-sm text-muted-foreground">
        {value === "keep"
          ? "Members with custom amounts keep their budgets"
          : "Matching reset periods only. Selected amounts follow future team defaults. Spend, reset periods, and temporary increases stay unchanged"}
        {(value === "raise" || value === "both") &&
          ". Includes $0 budgets, which may allow those members to spend again"}
      </p>
    </div>
  );
}
