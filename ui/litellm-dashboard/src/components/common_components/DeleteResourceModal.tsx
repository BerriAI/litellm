import { CircleAlert } from "lucide-react";
import React, { useState, useEffect } from "react";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";

interface DeleteResourceModalProps {
  isOpen: boolean;
  title: string;
  alertMessage?: string;
  message: string;
  resourceInformationTitle?: string;
  resourceInformation?: Array<{
    label: string;
    value: string | number | undefined | null;
    code?: boolean;
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
  const [requiredConfirmationInput, setRequiredConfirmationInput] = useState("");

  useEffect(() => {
    if (isOpen) {
      setRequiredConfirmationInput("");
    }
  }, [isOpen]);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && !confirmLoading && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {alertMessage && (
            <Alert variant="warning">
              <AlertTitle>{alertMessage}</AlertTitle>
            </Alert>
          )}
          <Card size="sm" className="mt-4">
            {resourceInformationTitle && (
              <CardHeader className="border-b">
                <CardTitle>{resourceInformationTitle}</CardTitle>
              </CardHeader>
            )}
            <CardContent>
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1">
                {resourceInformation?.map(({ label, value, code }) => (
                  <React.Fragment key={label}>
                    <dt className="font-semibold">{label}</dt>
                    <dd className="min-w-0 break-words">{code ? <code>{value ?? "-"}</code> : value ?? "-"}</dd>
                  </React.Fragment>
                ))}
              </dl>
            </CardContent>
          </Card>
          <div>
            <span>{message}</span>
          </div>
          {requiredConfirmation && (
            <div className="mb-6 mt-4 pt-4 border-t border-border">
              <p className="block text-base font-medium text-foreground mb-2">
                Type <span className="font-semibold text-destructive">{requiredConfirmation}</span> to confirm deletion:
              </p>
              <InputGroup className="rounded-md">
                <InputGroupAddon>
                  <CircleAlert className="size-3.5 text-destructive" />
                </InputGroupAddon>
                <InputGroupInput
                  value={requiredConfirmationInput}
                  onChange={(e) => setRequiredConfirmationInput(e.target.value)}
                  placeholder={requiredConfirmation}
                  autoFocus
                />
              </InputGroup>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={confirmLoading}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onOk}
            disabled={(!!requiredConfirmation && requiredConfirmationInput !== requiredConfirmation) || confirmLoading}
          >
            {confirmLoading ? "Deleting..." : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
