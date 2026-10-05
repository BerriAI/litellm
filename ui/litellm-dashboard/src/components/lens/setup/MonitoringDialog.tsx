import { useState } from "react";
import { z } from "zod";
import { Controller } from "react-hook-form";
import { useZodForm } from "@/lib/forms/useZodForm";
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

const monitoringSchema = z.object({
  interval_minutes: z.number().int().min(1),
});

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
  const [error, setError] = useState("");
  const form = useZodForm(monitoringSchema, {
    defaultValues: { interval_minutes: settings.interval_minutes ?? 30 },
    mode: "onChange",
  });
  const { formState, control } = form;
  const save = form.handleSubmit(async ({ interval_minutes }) => {
    setError("");
    try {
      await onSave({ ...settings, enabled: true, interval_minutes });
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not enable monitoring");
    }
  });
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !formState.isSubmitting) onClose();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-xl">Keep monitoring</DialogTitle>
          <DialogDescription>
            Repeat this investigation with the saved scope, sample, model, and budget.
          </DialogDescription>
        </DialogHeader>
        <Controller
          control={control}
          name="interval_minutes"
          render={({ field }) => (
            <DurationInput label="Check every" value={field.value} onChange={field.onChange} base="minutes" />
          )}
        />
        {formState.errors.interval_minutes?.message && (
          <p role="alert" className="text-sm text-destructive">
            {formState.errors.interval_minutes.message}
          </p>
        )}
        <p className="text-xs leading-5 text-muted-foreground">
          Each investigation looks back over the saved time range. The interval starts after the previous run finishes.
        </p>
        {!ready && (
          <p role="status" className="text-sm text-warning">
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
          <Button disabled={!formState.isValid || formState.isSubmitting || !ready} onClick={() => void save()}>
            {formState.isSubmitting ? "Saving…" : "Enable monitoring"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
