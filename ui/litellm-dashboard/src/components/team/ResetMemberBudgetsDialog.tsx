import React from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface ResetMemberBudgetsDialogProps {
  open: boolean;
  memberCount: number;
  newBudget: number;
  applying: boolean;
  onResetAll: () => void;
  onKeepCustom: () => void;
  onCancel: () => void;
}

export default function ResetMemberBudgetsDialog({
  open,
  memberCount,
  newBudget,
  applying,
  onResetAll,
  onKeepCustom,
  onCancel,
}: ResetMemberBudgetsDialogProps) {
  const plural = memberCount !== 1;
  return (
    <Dialog open={open} onOpenChange={(nextOpen) => !nextOpen && !applying && onCancel()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reset member budgets?</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          {memberCount} {plural ? "members have" : "member has"} a custom budget, so the new team default of $
          {newBudget} will not apply to {plural ? "them" : "that member"}. Reset {plural ? "them" : "it"} to the
          default, or keep the custom {plural ? "budgets" : "budget"}?
        </p>
        <DialogFooter>
          <Button variant="ghost" onClick={onCancel} disabled={applying}>
            Cancel
          </Button>
          <Button variant="outline" onClick={onKeepCustom} disabled={applying}>
            Keep custom {plural ? "budgets" : "budget"}
          </Button>
          <Button onClick={onResetAll} disabled={applying}>
            {plural ? "Reset all" : "Reset"} to ${newBudget}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
