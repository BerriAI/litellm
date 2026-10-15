import { useEffect } from "react";

import {
  useTernaryCreate,
  useTernaryUpdate,
  type TernarySettingsView,
} from "@/app/(dashboard)/hooks/ternary/useTernary";
import { FormField } from "@/components/shared/form/FormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

import {
  buildCreatePayload,
  buildUpdatePayload,
  createSchema,
  EMPTY_TERNARY_FORM_VALUES,
  formValuesFromSettings,
  FREQUENCY_ITEMS,
  updateSchema,
  type TernaryFormValues,
} from "./ternaryPayload";

interface TernarySettingsModalProps {
  open: boolean;
  settings: TernarySettingsView | null;
  onSaved: (message: string) => void;
  onCancel: () => void;
}

export default function TernarySettingsModal({ open, settings, onSaved, onCancel }: TernarySettingsModalProps) {
  const isEdit = settings !== null;
  const form = useZodForm(isEdit ? updateSchema : createSchema, { defaultValues: EMPTY_TERNARY_FORM_VALUES });
  const createMutation = useTernaryCreate();
  const updateMutation = useTernaryUpdate();
  const isPending = createMutation.isPending || updateMutation.isPending;
  const frequency = form.watch("frequency");
  const submitLabel = isEdit ? "Update" : "Create";

  useEffect(() => {
    if (open) {
      form.reset(settings ? formValuesFromSettings(settings) : EMPTY_TERNARY_FORM_VALUES);
    }
  }, [open, settings, form]);

  const callbacks = {
    onSuccess: (response: { message: string }) => onSaved(response.message),
    onError: (error: Error) => toast.error(error.message || "Failed to save the Ternary integration"),
  };

  const handleSubmit = (values: TernaryFormValues) => {
    if (isEdit) {
      updateMutation.mutate(buildUpdatePayload(values), callbacks);
    } else {
      createMutation.mutate(buildCreatePayload(values), callbacks);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{isEdit ? "Edit Ternary Integration" : "Add Ternary Integration"}</DialogTitle>
          <DialogDescription>
            Create an AI Gateway integration in Ternary (Admin, Integrations, New Integration) to get a connection
            ID and API key.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="api_key"
              label="API Key"
              description={isEdit ? "Leave empty to keep the current key" : undefined}
            >
              {({ ref, ...field }) => (
                <PasswordInput
                  {...field}
                  ref={ref}
                  autoComplete="off"
                  placeholder={isEdit ? "Leave empty to keep existing" : "Enter your Ternary API key"}
                />
              )}
            </FormField>
            <FormField control={form.control} name="connection_id" label="Connection ID">
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="Enter your Ternary connection ID" />}
            </FormField>
            <FormField control={form.control} name="base_url" label="Base URL">
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="https://api.ternary.app" />}
            </FormField>
            <FormField control={form.control} name="frequency" label="Export Frequency">
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": describedBy }) => (
                <Select items={FREQUENCY_ITEMS} value={value} onValueChange={onChange}>
                  <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={describedBy} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {FREQUENCY_ITEMS.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>
            {frequency === "interval" && (
              <FormField control={form.control} name="interval_seconds" label="Interval (seconds)">
                {({ ref, ...field }) => <Input {...field} ref={ref} inputMode="numeric" placeholder="300" />}
              </FormField>
            )}
          </FieldGroup>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button onClick={() => void form.handleSubmit(handleSubmit)()} disabled={isPending} aria-busy={isPending}>
            {isPending ? "Saving..." : submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
