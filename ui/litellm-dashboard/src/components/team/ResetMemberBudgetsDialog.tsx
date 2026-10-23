import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldLabel } from "@/components/ui/field";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import {
  pluralize,
  type MemberBudgetBackfill,
  type MemberBudgetBackfillPreview,
  type MemberBudgetResetPending,
} from "./memberBudgetReset";
import type { MemberBudgetResetState } from "./useMemberBudgetReset";

interface ResetMemberBudgetsDialogProps {
  state: MemberBudgetResetState;
  onSave: (backfill: MemberBudgetBackfill) => void;
  onDismiss: () => void;
}

interface BackfillOption {
  value: MemberBudgetBackfill;
  label: string;
}

const members = (count: number) => `${count} ${pluralize(count, "member", "members")}`;

export const backfillOptions = (preview: MemberBudgetBackfillPreview, newBudget: number): BackfillOption[] => {
  const amount = `$${formatNumberWithCommas(newBudget, 2)}`;
  const options: (BackfillOption | null)[] = [
    { value: "none", label: "Keep all custom budgets" },
    preview.below > 0 ? { value: "raise_to_default", label: `Raise ${members(preview.below)} below ${amount}` } : null,
    preview.above > 0 ? { value: "lower_to_default", label: `Lower ${members(preview.above)} above ${amount}` } : null,
    preview.below > 0 && preview.above > 0
      ? { value: "all_to_default", label: `Raise and lower all ${members(preview.below + preview.above)}` }
      : null,
  ];
  return options.filter((option): option is BackfillOption => option !== null);
};

interface BackfillChoiceProps {
  pending: MemberBudgetResetPending;
  busy: boolean;
  onSave: (backfill: MemberBudgetBackfill) => void;
  onDismiss: () => void;
}

function BackfillChoice({ pending, busy, onSave, onDismiss }: BackfillChoiceProps) {
  const [choice, setChoice] = useState<MemberBudgetBackfill>("none");
  const otherResetPeriod = pending.preview.otherResetPeriod;

  return (
    <>
      <DialogHeader>
        <DialogTitle>Apply the ${formatNumberWithCommas(pending.newBudget, 2)} default to custom budgets?</DialogTitle>
      </DialogHeader>
      <RadioGroup value={choice} onValueChange={(value) => setChoice(value as MemberBudgetBackfill)}>
        {backfillOptions(pending.preview, pending.newBudget).map((option) => (
          <FieldLabel key={option.value} className="font-normal">
            <RadioGroupItem value={option.value} disabled={busy} />
            {option.label}
          </FieldLabel>
        ))}
      </RadioGroup>
      {otherResetPeriod > 0 && (
        <p className="text-sm text-muted-foreground">
          {members(otherResetPeriod)} {pluralize(otherResetPeriod, "resets", "reset")} on a different schedule and
          won&apos;t change
        </p>
      )}
      <DialogFooter>
        <Button variant="ghost" disabled={busy} onClick={onDismiss}>
          Cancel
        </Button>
        <Button disabled={busy} onClick={() => onSave(choice)}>
          Save
        </Button>
      </DialogFooter>
    </>
  );
}

export default function ResetMemberBudgetsDialog({ state, onSave, onDismiss }: ResetMemberBudgetsDialogProps) {
  const busy = state.phase === "saving";

  return (
    <Dialog
      open={state.phase !== "idle"}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !busy) onDismiss();
      }}
    >
      <DialogContent>
        {state.phase !== "idle" && (
          <BackfillChoice pending={state.pending} busy={busy} onSave={onSave} onDismiss={onDismiss} />
        )}
      </DialogContent>
    </Dialog>
  );
}
