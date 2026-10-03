import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DurationInput } from "@/components/shared/DurationInput";
import { type Settings } from "../model/types";

export function MonitoringDialog({
  settings,
  ready,
  onSave,
  onClose,
}: {
  settings: Settings;
  ready: boolean;
  onSave: (settings: Settings) => Promise<void>;
  onClose: () => void;
}) {
  const [interval, setInterval] = useState(settings.interval_minutes ?? 30);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const save = async () => {
    setBusy(true);
    try {
      await onSave({ ...settings, enabled: true, interval_minutes: interval });
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not enable monitoring");
    } finally {
      setBusy(false);
    }
  };
  const validInterval = Number.isInteger(interval) && interval >= 1 && interval <= 10080;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-xl">Keep monitoring</DialogTitle>
          <DialogDescription>
            Repeat this investigation with the saved scope, sample, model, and budget.
          </DialogDescription>
        </DialogHeader>
        <DurationInput label="Check every" value={interval} onChange={setInterval} base="minutes" max={10080} />
        <p className="text-xs leading-5 text-muted-foreground">
          Each investigation looks back over the saved time range. The interval starts after the previous run finishes.
        </p>
        {!ready && (
          <p role="status" className="text-sm text-amber-700">
            Reconnect the worker before enabling monitoring.
          </p>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button disabled={busy || !ready || !validInterval} onClick={() => void save()}>
            {busy ? "Saving…" : "Enable monitoring"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
