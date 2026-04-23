import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { AlertCircle } from "lucide-react";
import React, { useState, useEffect } from "react";

interface DeleteResourceModalProps {
  isOpen: boolean;
  title: string;
  alertMessage?: string;
  message: string;
  resourceInformationTitle?: string;
  resourceInformation?: Array<{
    label: string;
    value: string | number | undefined | null;
    // Kept for back-compat with callers that used to spread antd
    // Typography.Text props (e.g. `type="danger"`, `strong`). Now only
    // `className` is applied.
    className?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
  }>;
  onCancel: () => void;
  onOk: () => void;
  confirmLoading: boolean;
  requiredConfirmation?: string;
}

export default function DeleteResourceModal({
  isOpen,
  title,
  alertMessage,
  message,
  resourceInformationTitle,
  resourceInformation,
  onCancel,
  onOk,
  confirmLoading,
  requiredConfirmation,
}: DeleteResourceModalProps) {
  const [requiredConfirmationInput, setRequiredConfirmationInput] =
    useState("");

  useEffect(() => {
    if (isOpen) setRequiredConfirmationInput("");
  }, [isOpen]);

  const confirmationBlocked =
    !!requiredConfirmation && requiredConfirmationInput !== requiredConfirmation;

  return (
    <AlertDialog
      open={isOpen}
      onOpenChange={(o) => (!o ? onCancel() : undefined)}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          {/* Hoist alert + resource info up so the AlertDialogDescription
              stays a plain string/paragraph (it must be a DIV-less element
              inside <p> to satisfy Radix/a11y). */}
        </AlertDialogHeader>

        <div className="space-y-4">
          {alertMessage && (
            <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-lg">
              <p className="text-sm text-amber-800 dark:text-amber-200">
                {alertMessage}
              </p>
            </div>
          )}
          {resourceInformationTitle || resourceInformation?.length ? (
            <Card className="bg-destructive/5 border-destructive/30 p-0">
              {resourceInformationTitle && (
                <div className="px-4 py-2 border-b border-destructive/30 bg-destructive/10 rounded-t-md">
                  <span className="font-medium">
                    {resourceInformationTitle}
                  </span>
                </div>
              )}
              <div className="p-4 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-sm">
                {resourceInformation?.map(({ label, value, className }) => (
                  <React.Fragment key={label}>
                    <span className="font-semibold">{label}</span>
                    <span className={className}>{value ?? "-"}</span>
                  </React.Fragment>
                ))}
              </div>
            </Card>
          ) : null}
          <AlertDialogDescription>{message}</AlertDialogDescription>
          {requiredConfirmation && (
            <div className="mb-2 mt-2 pt-3 border-t border-border">
              <div className="block text-base font-medium text-foreground mb-2">
                Type{" "}
                <span className="font-bold text-destructive">
                  {requiredConfirmation}
                </span>{" "}
                to confirm deletion:
              </div>
              <div className="relative">
                <AlertCircle className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-destructive pointer-events-none" />
                <Input
                  value={requiredConfirmationInput}
                  onChange={(e) =>
                    setRequiredConfirmationInput(e.target.value)
                  }
                  placeholder={requiredConfirmation}
                  className="pl-7"
                  autoFocus
                />
              </div>
            </div>
          )}
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={confirmLoading}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            disabled={confirmationBlocked || confirmLoading}
            onClick={(e) => {
              e.preventDefault();
              if (!confirmationBlocked) onOk();
            }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {confirmLoading ? "Deleting..." : "Delete"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
