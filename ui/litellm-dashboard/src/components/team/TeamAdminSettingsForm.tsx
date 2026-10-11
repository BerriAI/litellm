"use client";

import { Save } from "lucide-react";
import { useWatch } from "react-hook-form";
import { z } from "zod";

import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { FieldGroup } from "@/components/ui/field";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

import NumericalInput from "../shared/numerical_input";
import {
  TEAM_ADMIN_SETTINGS_FIELDS,
  teamAdminFieldLabel,
  teamAdminSettingsChanges,
  type TeamAdminSettingsChanges,
  type TeamAdminSettingsField,
  type TeamAdminSettingsValues,
} from "./teamAdminEditAccess";

const numericInputSchema = z.union([z.string(), z.number()]).nullish();

const teamAdminSettingsSchema = z.object({
  tpm_limit: numericInputSchema,
  rpm_limit: numericInputSchema,
  max_budget: numericInputSchema,
});

const INPUT_STEP: Readonly<Record<TeamAdminSettingsField, number>> = { tpm_limit: 1, rpm_limit: 1, max_budget: 0.01 };

const MAX_BUDGET_KEEP_OR_LOWER_HINT = "You can keep or lower this budget. Ask a proxy admin to raise it.";
const MAX_BUDGET_RAISE_HINT =
  "You can raise this budget. Raises are capped by your organization's budget when the team belongs to one.";

const fieldHint = (name: TeamAdminSettingsField, mayRaiseMaxBudget: boolean): string | undefined => {
  if (name !== "max_budget") return undefined;
  return mayRaiseMaxBudget ? MAX_BUDGET_RAISE_HINT : MAX_BUDGET_KEEP_OR_LOWER_HINT;
};

interface TeamAdminSettingsFormProps {
  initialValues: TeamAdminSettingsValues;
  editableFields: ReadonlySet<string>;
  mayRaiseMaxBudget: boolean;
  isSaving: boolean;
  onCancel: () => void;
  onSave: (changes: TeamAdminSettingsChanges) => Promise<void>;
}

export default function TeamAdminSettingsForm({
  initialValues,
  editableFields,
  mayRaiseMaxBudget,
  isSaving,
  onCancel,
  onSave,
}: TeamAdminSettingsFormProps) {
  const form = useZodForm(teamAdminSettingsSchema, { defaultValues: initialValues });
  const draft = useWatch({ control: form.control });
  const hasChanges = Object.keys(teamAdminSettingsChanges(draft, initialValues, editableFields)).length > 0;
  const submit = form.handleSubmit((values) => onSave(teamAdminSettingsChanges(values, initialValues, editableFields)));

  return (
    <form onSubmit={(event) => void submit(event)}>
      <FieldGroup>
        <p className="text-sm text-muted-foreground">
          A proxy admin chose which settings team admins can change. Ask a proxy admin to change anything else.
        </p>
        {TEAM_ADMIN_SETTINGS_FIELDS.filter((name) => editableFields.has(name)).map((name) => (
          <FormField
            key={name}
            control={form.control}
            name={name}
            label={teamAdminFieldLabel(name)}
            description={fieldHint(name, mayRaiseMaxBudget)}
          >
            {({ ref, value, ...field }) => (
              <NumericalInput {...field} ref={ref} value={value ?? ""} step={INPUT_STEP[name]} />
            )}
          </FormField>
        ))}
      </FieldGroup>

      <div className="mt-6 flex items-center justify-end gap-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={isSaving}>
          Cancel
        </Button>
        <Button type="submit" disabled={isSaving || !hasChanges}>
          {isSaving ? <UiLoadingSpinner className="size-4" /> : <Save className="size-4" />}
          Save Changes
        </Button>
      </div>
    </form>
  );
}
