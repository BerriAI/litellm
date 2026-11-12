"use client";

import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  return (
    <Dialog open={open} onOpenChange={(next) => onOpenChange(next)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {premiumUser ? t("telemetry.enterpriseDialog.premiumTitle") : t("telemetry.enterpriseDialog.title")}
          </DialogTitle>
          <DialogDescription>
            {premiumUser
              ? t("telemetry.enterpriseDialog.premiumDescription")
              : t("telemetry.enterpriseDialog.description")}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter showCloseButton />
      </DialogContent>
    </Dialog>
  );
}
