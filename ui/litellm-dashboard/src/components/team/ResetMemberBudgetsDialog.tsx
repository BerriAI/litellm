import { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { MemberBudgetResetPending, MemberBudgetUpdateMode } from "./memberBudgetReset";

const OPTIONS: Record<MemberBudgetUpdateMode, string> = {
  keep: "Keep custom budgets",
  raise: "Raise smaller budgets",
  lower: "Lower larger budgets",
  both: "Raise and lower",
};

interface ResetMemberBudgetsDialogProps {
  pending: MemberBudgetResetPending;
  busy: boolean;
  onSave: (mode: MemberBudgetUpdateMode) => void;
  onDismiss: () => void;
}

export default function ResetMemberBudgetsDialog({ pending, busy, onSave, onDismiss }: ResetMemberBudgetsDialogProps) {
  const [mode, setMode] = useState<MemberBudgetUpdateMode>("keep");
  const id = useId();

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onDismiss();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reset member budgets?</DialogTitle>
          <DialogDescription>
            {pending.memberCount} {pending.memberCount === 1 ? "member has" : "members have"} a custom budget. Choose
            which amounts should follow the new ${formatNumberWithCommas(pending.newBudget, 2)} team default
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <label htmlFor={id} className="text-sm font-medium">
            Existing member budgets
          </label>
          <Select<MemberBudgetUpdateMode>
            items={OPTIONS}
            value={mode}
            onValueChange={(next) => {
              if (next !== null) setMode(next);
            }}
            disabled={busy}
          >
            <SelectTrigger id={id} aria-describedby={`${id}-description`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(OPTIONS).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p id={`${id}-description`} className="text-sm text-muted-foreground">
            {mode === "keep"
              ? "Custom amounts stay unchanged"
              : "Only different amounts with matching reset periods change. Selected amounts follow future team defaults"}
            {(mode === "raise" || mode === "both") && ". Includes $0 budgets, which may allow spending again"}
          </p>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onDismiss} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={() => onSave(mode)} disabled={busy}>
            {busy ? "Saving..." : "Save changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
