"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as React from "react";
import { useTranslation } from "react-i18next";

import { organizationKeys } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";

import { NO_RESET } from "../org-settings/OrgSettingsForm";
import { orgSettingsSchema } from "../org-settings/schema";
import { buildOrgCreateBody, emptyOrgFormValues, type OrgCreateBody } from "./mapper";

const defaultCreateOrganization = async (body: OrgCreateBody): Promise<unknown> => {
  const { data } = await fetchClient.POST("/organization/new", { body });
  return data;
};

interface OrgCreateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  accessToken: string;
  createOrganization?: (body: OrgCreateBody) => Promise<unknown>;
}

export const OrgCreateDialog = ({
  open,
  onOpenChange,
  accessToken,
  createOrganization = defaultCreateOrganization,
}: OrgCreateDialogProps) => {
  const { t } = useTranslation("gateway");
  const queryClient = useQueryClient();
  const form = useZodForm(orgSettingsSchema, { defaultValues: emptyOrgFormValues });
  const budgetDurationOptions = [
    { value: NO_RESET, label: t("organizations.form.noReset") },
    { value: "24h", label: t("budgets.duration.daily") },
    { value: "7d", label: t("budgets.duration.weekly") },
    { value: "30d", label: t("budgets.duration.monthly") },
  ];

  const closeAndReset = () => {
    form.reset(emptyOrgFormValues);
    onOpenChange(false);
  };

  const mutation = useMutation({
    mutationFn: (body: OrgCreateBody) => createOrganization(body),
    onSuccess: () => {
      NotificationsManager.success(t("organizations.notifications.created"));
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
      closeAndReset();
    },
    onError: (error: unknown) =>
      NotificationsManager.fromBackend(
        error instanceof Error ? error.message : t("organizations.notifications.createFailed"),
      ),
  });

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen && mutation.isPending) return;
    if (!nextOpen) {
      form.reset(emptyOrgFormValues);
    }
    onOpenChange(nextOpen);
  };

  const onSubmit = form.handleSubmit((values) => {
    if (mutation.isPending) return;
    mutation.mutate(buildOrgCreateBody(values));
  });

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("organizations.form.createTitle")}</DialogTitle>
        </DialogHeader>

        <form onSubmit={onSubmit} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="organization_alias"
              label={t("organizations.form.organizationName")}
            >
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

            <FormField
              control={form.control}
              name="vector_stores"
              label={t("organizations.form.allowedVectorStores")}
              description={t("organizations.form.vectorStoresDescription")}
            >
              {(field) => (
                <VectorStoreSelector
                  value={field.value}
                  onChange={field.onChange}
                  accessToken={accessToken}
                  placeholder={t("organizations.form.selectVectorStoresOptional")}
                />
              )}
            </FormField>

            <FormField
              control={form.control}
              name="mcp"
              label={t("organizations.form.allowedMcp")}
              description={t("organizations.form.mcpDescription")}
            >
              {(field) => (
                <MCPServerSelector
                  value={field.value}
                  onChange={field.onChange}
                  accessToken={accessToken}
                  placeholder={t("organizations.form.selectMcpOptional")}
                />
              )}
            </FormField>

            <FormField control={form.control} name="metadata" label={t("organizations.form.metadata")}>
              {({ ref, ...field }) => <Textarea {...field} ref={ref} rows={4} />}
            </FormField>
          </FieldGroup>

          <DialogFooter className="mt-6">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              disabled={mutation.isPending}
            >
              {t("organizations.form.cancel")}
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? t("organizations.form.creating") : t("organizations.form.create")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
