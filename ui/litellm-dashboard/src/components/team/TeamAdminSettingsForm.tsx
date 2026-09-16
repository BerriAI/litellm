"use client";

import { Save } from "lucide-react";
import { z } from "zod/v4";

import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { FieldGroup } from "@/components/ui/field";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

import NumericalInput from "../shared/numerical_input";
import {
  teamAdminFieldLabel,
  teamAdminSettingsChanges,
  type TeamAdminSettingsChanges,
  type TeamAdminSettingsValues,
} from "./teamAdminEditAccess";

const teamAdminSettingsSchema = z.object({
  tpm_limit: z.union([z.string(), z.number()]).nullish(),
});

interface TeamAdminSettingsFormProps {
  initialValues: TeamAdminSettingsValues;
  editableFields: ReadonlySet<string>;
  isSaving: boolean;
  onCancel: () => void;
  onSave: (changes: TeamAdminSettingsChanges) => Promise<void>;
}

export default function TeamAdminSettingsForm({
  initialValues,
  editableFields,
  isSaving,
  onCancel,
  onSave,
}: TeamAdminSettingsFormProps) {
  const form = useZodForm(teamAdminSettingsSchema, { defaultValues: initialValues });
  const submit = form.handleSubmit((values) => onSave(teamAdminSettingsChanges(values, editableFields)));

  return (
    <form onSubmit={(event) => void submit(event)}>
      <FieldGroup>
        <p className="text-sm text-muted-foreground">
          A proxy admin chose which settings team admins can change. Ask a proxy admin to change anything else.
        </p>
        {editableFields.has("tpm_limit") && (
          <FormField control={form.control} name="tpm_limit" label={teamAdminFieldLabel("tpm_limit")}>
            {({ ref, value, ...field }) => <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} />}
          </FormField>
        )}
      </FieldGroup>

      <div className="mt-6 flex items-center justify-end gap-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={isSaving}>
          Cancel
        </Button>
        <Button type="submit" disabled={isSaving}>
          {isSaving ? <UiLoadingSpinner className="size-4" /> : <Save className="size-4" />}
          Save Changes
        </Button>
      </div>
    </form>
  );
}
