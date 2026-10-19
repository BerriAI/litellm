"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

type CustomerBlockConfirmationProps = {
  open: boolean;
  count: number;
  blocked: boolean;
  isPending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

export function CustomerBlockConfirmation({
  open,
  count,
  blocked,
  isPending,
  onCancel,
  onConfirm,
}: CustomerBlockConfirmationProps) {
  const action = blocked ? "Block" : "Unblock";
  const customerLabel = count === 1 ? "this customer" : "these customers";
  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !isPending) onCancel();
      }}
    >
      <DialogContent showCloseButton={!isPending}>
        <DialogHeader>
          <DialogTitle>
            {action} {count} customer{count === 1 ? "" : "s"}?
          </DialogTitle>
          <DialogDescription>
            {blocked
              ? `Requests for ${customerLabel} will be rejected until unblocked. Existing settings and permissions are kept.`
              : `Requests for ${customerLabel} will be allowed again, subject to existing limits and permissions.`}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button variant={blocked ? "destructive" : "default"} onClick={onConfirm} disabled={isPending}>
            {action}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
