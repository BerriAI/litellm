"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { ParseKeys, TFunction } from "i18next";
import * as React from "react";
import { useTranslation } from "react-i18next";

import { organizationKeys } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import { toast } from "@/lib/toast";
import type { Organization } from "@/components/networking";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";
import { pickDirty } from "@/lib/forms/pickDirty";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";

import { buildOrgPatch, orgToForm, type OrgPatchBody } from "./mapper";
import { buildOrgSettingsSchema } from "./schema";

export const NO_RESET = "never";

const BUDGET_DURATION_OPTIONS: readonly { value: string; labelKey: ParseKeys }[] = [
  { value: NO_RESET, labelKey: "organizations.settingsForm.budgetDuration.noReset" },
  { value: "24h", labelKey: "organizations.settingsForm.budgetDuration.daily" },
  { value: "7d", labelKey: "organizations.settingsForm.budgetDuration.weekly" },
  { value: "30d", labelKey: "organizations.settingsForm.budgetDuration.monthly" },
];

export const budgetDurationItems = (t: TFunction): { value: string; label: string }[] =>
  BUDGET_DURATION_OPTIONS.map((option) => ({ value: option.value, label: t(option.labelKey) }));

const defaultPatchOrganization = async (organizationId: string, body: OrgPatchBody): Promise<unknown> => {
  const { data } = await fetchClient.PATCH("/v2/organization/{organization_id}", {
    params: { path: { organization_id: organizationId } },
    body,
  });
  return data;
};

interface OrgSettingsFormProps {
  organizationId: string;
  org: Organization;
  accessToken: string;
  onCancel: () => void;
  onSaved: () => void;
  patchOrganization?: (organizationId: string, body: OrgPatchBody) => Promise<unknown>;
}

export const OrgSettingsForm = ({
  organizationId,
  org,
  accessToken,
  onCancel,
  onSaved,
  patchOrganization = defaultPatchOrganization,
}: OrgSettingsFormProps) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const schema = React.useMemo(() => buildOrgSettingsSchema(t), [t]);
  const form = useZodForm(schema, { defaultValues: orgToForm(org) });
  const budgetDurationOptions = budgetDurationItems(t);
  const { isDirty } = form.formState;

  const mutation = useMutation({
    mutationFn: (body: OrgPatchBody) => patchOrganization(organizationId, body),
    onSuccess: () => {
      toast.success(t("organizations.settingsForm.toasts.updated"));
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
      onSaved();
    },
    onError: (error: unknown) =>
      toast.fromError(error instanceof Error ? error.message : t("organizations.settingsForm.toasts.updateFailed")),
  });

  const onSubmit = form.handleSubmit((values) => {
    mutation.mutate(buildOrgPatch(pickDirty(values, form.formState.dirtyFields)));
  });

  return (
    <form onSubmit={onSubmit} noValidate>
      <FieldGroup>
        <FormField control={form.control} name="organization_alias" label={t("organizations.settingsForm.fields.name")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} />}
        </FormField>

        <FormField control={form.control} name="models" label={t("organizations.settingsForm.fields.models")}>
          {(field) => (
            <ModelSelect
              value={field.value}
              onChange={field.onChange}
              context="organization"
              options={{ includeSpecialOptions: true, showAllProxyModelsOverride: true }}
            />
          )}
        </FormField>

        <FormField control={form.control} name="max_budget" label={t("organizations.settingsForm.fields.maxBudget")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step="any" min={0} />}
        </FormField>

        <FormField
          control={form.control}
          name="budget_duration"
          label={t("organizations.settingsForm.fields.budgetDuration")}
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Select
              items={budgetDurationOptions}
              value={value === "" ? NO_RESET : value}
              onValueChange={(selected) => onChange(selected === NO_RESET ? "" : selected)}
            >
              <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {budgetDurationOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </FormField>

        <FormField control={form.control} name="tpm_limit" label={t("organizations.settingsForm.fields.tpmLimit")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
        </FormField>

        <FormField control={form.control} name="rpm_limit" label={t("organizations.settingsForm.fields.rpmLimit")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
        </FormField>

        <FormField
          control={form.control}
          name="vector_stores"
          label={t("organizations.settingsForm.fields.vectorStores")}
        >
          {(field) => (
            <VectorStoreSelector
              value={field.value}
              onChange={field.onChange}
              accessToken={accessToken}
              placeholder={t("organizations.settingsForm.fields.vectorStoresPlaceholder")}
            />
          )}
        </FormField>

        <FormField control={form.control} name="mcp" label={t("organizations.settingsForm.fields.mcp")}>
          {(field) => (
            <MCPServerSelector
              value={field.value}
              onChange={field.onChange}
              accessToken={accessToken}
              placeholder={t("organizations.settingsForm.fields.mcpPlaceholder")}
            />
          )}
        </FormField>

        <FormField control={form.control} name="metadata" label={t("organizations.settingsForm.fields.metadata")}>
          {({ ref, ...field }) => <Textarea {...field} ref={ref} rows={4} />}
        </FormField>
      </FieldGroup>

      <div className="sticky z-chrome bg-card p-4 border-t border-border -bottom-6 -inset-x-6 mt-6">
        <div className="flex justify-end items-center gap-2">
          <Button type="button" variant="outline" onClick={onCancel} disabled={mutation.isPending}>
            {t("organizations.settingsForm.cancel")}
          </Button>
          <Button type="submit" disabled={!isDirty || mutation.isPending}>
            {mutation.isPending ? t("organizations.settingsForm.saving") : t("organizations.settingsForm.save")}
          </Button>
        </div>
      </div>
    </form>
  );
};
