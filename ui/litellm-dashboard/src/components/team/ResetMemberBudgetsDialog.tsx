import React from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { pluralize } from "./memberBudgetReset";
import type { MemberBudgetResetState } from "./useMemberBudgetReset";

interface ResetMemberBudgetsDialogProps {
  state: MemberBudgetResetState;
  onReset: () => void;
  onRetry: () => void;
  onKeep: () => void;
  onDismiss: () => void;
}

export default function ResetMemberBudgetsDialog({
  state,
  onReset,
  onRetry,
  onKeep,
  onDismiss,
}: ResetMemberBudgetsDialogProps) {
  const open = state.phase !== "idle";
  const busy = state.phase === "resetting";
  const failed = state.phase === "resetFailed";
  const memberCount = state.phase === "idle" ? 0 : state.pending.userIds.length;
  const newBudget = state.phase === "idle" ? 0 : state.pending.newBudget;

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !busy) onDismiss();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reset member budgets?</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          {memberCount} {pluralize(memberCount, "member has", "members have")} a custom budget, so the new team
          default of ${formatNumberWithCommas(newBudget, 2)} will not apply to{" "}
          {pluralize(memberCount, "that member", "them")}. Reset {pluralize(memberCount, "it", "them")} to the
          default, or keep the custom {pluralize(memberCount, "budget", "budgets")}?
        </p>
        <DialogFooter>
          {failed ? (
            <>
              <Button variant="outline" onClick={onDismiss}>
                Cancel
              </Button>
              <Button onClick={onRetry}>Retry reset</Button>
            </>
          ) : (
            <>
              <Button variant="ghost" onClick={onDismiss} disabled={busy}>
                Cancel
              </Button>
              <Button variant="outline" onClick={onKeep} disabled={busy}>
                Keep custom {pluralize(memberCount, "budget", "budgets")}
              </Button>
              <Button onClick={onReset} disabled={busy}>
                {pluralize(memberCount, "Reset", "Reset all")} to ${formatNumberWithCommas(newBudget, 2)}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
