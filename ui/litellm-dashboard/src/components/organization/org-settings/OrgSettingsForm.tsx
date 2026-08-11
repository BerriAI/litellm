"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as React from "react";
import { useTranslation } from "react-i18next";

import { organizationKeys } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import NotificationsManager from "@/components/molecules/notifications_manager";
import type { Organization } from "@/components/networking";
import { FieldGroup } from "@/components/shared/form/field";
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
import { orgSettingsSchema } from "./schema";

export const NO_RESET = "never";

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
  const { t } = useTranslation("gateway");
  const queryClient = useQueryClient();
  const form = useZodForm(orgSettingsSchema, { defaultValues: orgToForm(org) });
  const { isDirty } = form.formState;
  const budgetDurationOptions = [
    { value: NO_RESET, label: t("organizations.form.noReset") },
    { value: "24h", label: t("budgets.duration.daily") },
    { value: "7d", label: t("budgets.duration.weekly") },
    { value: "30d", label: t("budgets.duration.monthly") },
  ];

  const mutation = useMutation({
    mutationFn: (body: OrgPatchBody) => patchOrganization(organizationId, body),
    onSuccess: () => {
      NotificationsManager.success(t("organizations.notifications.settingsUpdated"));
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
      onSaved();
    },
    onError: (error: unknown) =>
      NotificationsManager.fromBackend(
        error instanceof Error ? error.message : t("organizations.notifications.settingsFailed"),
      ),
  });

  const onSubmit = form.handleSubmit((values) => {
    mutation.mutate(buildOrgPatch(pickDirty(values, form.formState.dirtyFields)));
  });

  return (
    <form onSubmit={onSubmit} noValidate>
      <FieldGroup>
        <FormField control={form.control} name="organization_alias" label={t("organizations.form.organizationName")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} />}
        </FormField>

        <FormField control={form.control} name="models" label={t("organizations.form.models")}>
          {(field) => (
            <ModelSelect
              value={field.value}
              onChange={field.onChange}
              context="organization"
              options={{ includeSpecialOptions: true, showAllProxyModelsOverride: true }}
            />
          )}
        </FormField>

        <FormField control={form.control} name="max_budget" label={t("organizations.form.maxBudget")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step="any" min={0} />}
        </FormField>

        <FormField control={form.control} name="budget_duration" label={t("organizations.form.resetBudget")}>
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

        <FormField control={form.control} name="tpm_limit" label={t("organizations.form.tpmLimit")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
        </FormField>

        <FormField control={form.control} name="rpm_limit" label={t("organizations.form.rpmLimit")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
        </FormField>

        <FormField control={form.control} name="vector_stores" label={t("organizations.form.vectorStores")}>
          {(field) => (
            <VectorStoreSelector
              value={field.value}
              onChange={field.onChange}
              accessToken={accessToken}
              placeholder={t("organizations.form.selectVectorStores")}
            />
          )}
        </FormField>

        <FormField control={form.control} name="mcp" label={t("organizations.form.mcpAndAccessGroups")}>
          {(field) => (
            <MCPServerSelector
              value={field.value}
              onChange={field.onChange}
              accessToken={accessToken}
              placeholder={t("organizations.form.selectMcp")}
            />
          )}
        </FormField>

        <FormField control={form.control} name="metadata" label={t("organizations.form.metadata")}>
          {({ ref, ...field }) => <Textarea {...field} ref={ref} rows={4} />}
        </FormField>
      </FieldGroup>

      <div className="sticky z-10 bg-white p-4 border-t border-gray-200 -bottom-6 -inset-x-6 mt-6">
        <div className="flex justify-end items-center gap-2">
          <Button type="button" variant="outline" onClick={onCancel} disabled={mutation.isPending}>
            {t("organizations.form.cancel")}
          </Button>
          <Button type="submit" disabled={!isDirty || mutation.isPending}>
            {mutation.isPending ? t("organizations.form.saving") : t("organizations.form.save")}
          </Button>
        </div>
      </div>
    </form>
  );
};
