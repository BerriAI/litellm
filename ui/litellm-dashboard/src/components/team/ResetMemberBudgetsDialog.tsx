import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { formatNumberWithCommas } from "@/utils/dataUtils";
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
  const { t } = useLanguage();
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
          <DialogTitle>{t("Reset member budgets?")}</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          {t(
            memberCount === 1
              ? "{count} member has a custom budget, so the new team default of ${budget} will not apply to that member. Reset it to the default, or keep the custom budget?"
              : "{count} members have a custom budget, so the new team default of ${budget} will not apply to them. Reset them to the default, or keep the custom budgets?",
            { count: memberCount, budget: formatNumberWithCommas(newBudget, 2) },
          )}
        </p>
        <DialogFooter>
          {failed ? (
            <>
              <Button variant="outline" onClick={onDismiss}>
                {t("Cancel")}
              </Button>
              <Button onClick={onRetry}>{t("Retry reset")}</Button>
            </>
          ) : (
            <>
              <Button variant="ghost" onClick={onDismiss} disabled={busy}>
                {t("Cancel")}
              </Button>
              <Button variant="outline" onClick={onKeep} disabled={busy}>
                {t(memberCount === 1 ? "Keep custom budget" : "Keep custom budgets")}
              </Button>
              <Button onClick={onReset} disabled={busy}>
                {t(memberCount === 1 ? "Reset to ${budget}" : "Reset all to ${budget}", {
                  budget: formatNumberWithCommas(newBudget, 2),
                })}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
