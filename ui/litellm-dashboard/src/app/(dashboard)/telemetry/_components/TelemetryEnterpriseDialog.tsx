"use client";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export function TelemetryEnterpriseDialog({
  open,
  onOpenChange,
  premiumUser,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  premiumUser: boolean;
}) {
  return (
    <Dialog open={open} onOpenChange={(next) => onOpenChange(next)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {premiumUser ? "Contact your enterprise representative" : "Telemetry is an enterprise-only feature"}
          </DialogTitle>
          <DialogDescription>
            {premiumUser
              ? "Telemetry can't be turned on from this page. Reach out to your LiteLLM enterprise representative to turn it on for this proxy."
              : "Telemetry is only available on LiteLLM Enterprise, and this proxy isn't on an enterprise license."}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter showCloseButton />
      </DialogContent>
    </Dialog>
  );
}
