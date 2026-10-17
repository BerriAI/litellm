"use client";

import { Save } from "lucide-react";
import { useState } from "react";
import { useWatch } from "react-hook-form";
import { z } from "zod";

import type { TeamMetadataField } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";

import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

import BudgetDurationDropdown, { NEVER_RESETS_BUDGET_DURATION } from "../common_components/budget_duration_dropdown";
import MetadataKeyValueFields, {
  type MetadataPair,
  metadataPairsSchema,
} from "../common_components/MetadataKeyValueFields";
import NumericalInput from "../shared/numerical_input";
import {
  isTeamAdminDurationSettingsField,
  TEAM_ADMIN_METADATA_FIELD,
  type TeamAdminMetadataContext,
  teamAdminMetadataChange,
  teamAdminMetadataPairIssue,
  TEAM_ADMIN_SETTINGS_FIELDS,
  teamAdminFieldLabel,
  teamAdminSettingsChanges,
  type TeamAdminSettingsChanges,
  type TeamAdminSettingsField,
  type TeamAdminSettingsValues,
} from "./teamAdminEditAccess";

const numericInputSchema = z.union([z.string(), z.number()]).nullish();

const teamAdminSettingsShape = {
  tpm_limit: numericInputSchema,
  rpm_limit: numericInputSchema,
  max_budget: numericInputSchema,
  soft_budget: numericInputSchema,
  team_member_budget: numericInputSchema,
  team_member_budget_duration: z.string().nullish(),
  team_member_tpm_limit: numericInputSchema,
  team_member_rpm_limit: numericInputSchema,
};

const teamAdminSettingsSchemaFor = (metadataContext: TeamAdminMetadataContext | undefined) =>
  z.object({
    ...teamAdminSettingsShape,
    metadata: metadataPairsSchema
      .superRefine((pairs, ctx) => {
        if (metadataContext === undefined) return;
        pairs.forEach((pair, index) => {
          const issue = teamAdminMetadataPairIssue(pair, metadataContext);
          if (issue) ctx.addIssue({ code: "custom", message: issue.message, path: [index, issue.path] });
        });
      })
      .optional(),
  });

export interface TeamAdminMetadataSettings {
  readonly pairs: readonly MetadataPair[];
  readonly context: TeamAdminMetadataContext;
  readonly schemaFields: readonly TeamMetadataField[];
  readonly schemaLoading: boolean;
}

const INPUT_STEP: Readonly<Record<Exclude<TeamAdminSettingsField, "team_member_budget_duration">, number>> = {
  tpm_limit: 1,
  rpm_limit: 1,
  max_budget: 0.01,
  soft_budget: 0.01,
  team_member_budget: 0.01,
  team_member_tpm_limit: 1,
  team_member_rpm_limit: 1,
};

const MAX_BUDGET_KEEP_OR_LOWER_HINT = "You can keep or lower this budget. Ask a proxy admin to raise it.";
const MAX_BUDGET_RAISE_HINT =
  "You can raise this budget. Raises are capped by your organization's budget when the team belongs to one.";

const fieldHint = (name: TeamAdminSettingsField, mayRaiseMaxBudget: boolean): string | undefined => {
  if (name !== "max_budget") return undefined;
  return mayRaiseMaxBudget ? MAX_BUDGET_RAISE_HINT : MAX_BUDGET_KEEP_OR_LOWER_HINT;
};

const durationSelectValue = (value: string | number | null | undefined): string | undefined => {
  if (value === null) return NEVER_RESETS_BUDGET_DURATION;
  return typeof value === "string" ? value : undefined;
};

interface TeamAdminSettingsFormProps {
  initialValues: TeamAdminSettingsValues;
  metadata?: TeamAdminMetadataSettings;
  editableFields: ReadonlySet<string>;
  mayRaiseMaxBudget: boolean;
  isSaving: boolean;
  onCancel: () => void;
  onSave: (changes: TeamAdminSettingsChanges) => Promise<void>;
}

export default function TeamAdminSettingsForm(props: TeamAdminSettingsFormProps) {
  const waitsForMetadataSchema =
    props.metadata?.schemaLoading === true && props.editableFields.has(TEAM_ADMIN_METADATA_FIELD);
  if (waitsForMetadataSchema) {
    return (
      <div className="flex justify-center py-6">
        <UiLoadingSpinner className="size-5" />
      </div>
    );
  }
  return <TeamAdminSettingsFormFields {...props} />;
}

function TeamAdminSettingsFormFields({
  initialValues,
  metadata,
  editableFields,
  mayRaiseMaxBudget,
  isSaving,
  onCancel,
  onSave,
}: TeamAdminSettingsFormProps) {
  const editsMetadata = metadata !== undefined && editableFields.has(TEAM_ADMIN_METADATA_FIELD);
  const [schema] = useState(() => teamAdminSettingsSchemaFor(editsMetadata ? metadata.context : undefined));
  const form = useZodForm(schema, { defaultValues: { ...initialValues, metadata: [...(metadata?.pairs ?? [])] } });
  const draft = useWatch({ control: form.control });
  const changesFrom = (values: typeof draft): TeamAdminSettingsChanges => ({
    ...teamAdminSettingsChanges(values, initialValues, editableFields),
    ...(editsMetadata ? teamAdminMetadataChange(values.metadata, metadata.context) : {}),
  });
  const hasChanges = Object.keys(changesFrom(draft)).length > 0;
  const submit = form.handleSubmit((values) => onSave(changesFrom(values)));

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
            {({ ref, id, value, onChange, ...field }) =>
              isTeamAdminDurationSettingsField(name) ? (
                <BudgetDurationDropdown
                  id={id}
                  showNeverResets
                  placeholder="Inherit team reset period"
                  value={durationSelectValue(value)}
                  onChange={(next) => onChange(next === NEVER_RESETS_BUDGET_DURATION ? null : next ?? undefined)}
                />
              ) : (
                <NumericalInput
                  {...field}
                  ref={ref}
                  id={id}
                  value={value ?? ""}
                  onChange={onChange}
                  step={INPUT_STEP[name]}
                />
              )
            }
          </FormField>
        ))}
        {editsMetadata && (
          <Field>
            <FieldLabel>{teamAdminFieldLabel(TEAM_ADMIN_METADATA_FIELD)}</FieldLabel>
            <MetadataKeyValueFields
              control={form.control}
              getValues={form.getValues}
              name="metadata"
              schemaFields={metadata.schemaFields}
              schemaLoading={metadata.schemaLoading}
            />
            <FieldDescription>
              {
                'A value that is valid JSON, like 3, true or {"region": "us"}, is saved as that type, anything else as text. Keys the proxy reads as settings stay with proxy admins and are kept as they are.'
              }
            </FieldDescription>
          </Field>
        )}
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
