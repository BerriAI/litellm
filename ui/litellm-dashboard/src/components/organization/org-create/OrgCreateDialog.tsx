"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as React from "react";
import { useTranslation } from "react-i18next";

import { organizationKeys } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";

import { budgetDurationItems, NO_RESET } from "../org-settings/OrgSettingsForm";
import { buildOrgSettingsSchema } from "../org-settings/schema";
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
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const schema = React.useMemo(() => buildOrgSettingsSchema(t), [t]);
  const form = useZodForm(schema, { defaultValues: emptyOrgFormValues });
  const budgetDurationOptions = budgetDurationItems(t);

  const closeAndReset = () => {
    form.reset(emptyOrgFormValues);
    onOpenChange(false);
  };

  const mutation = useMutation({
    mutationFn: (body: OrgCreateBody) => createOrganization(body),
    onSuccess: () => {
      toast.success(t("organizations.createDialog.toasts.created"));
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
      closeAndReset();
    },
    onError: (error: unknown) =>
      toast.fromError(error instanceof Error ? error.message : t("organizations.createDialog.toasts.createFailed")),
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
          <DialogTitle>{t("organizations.createDialog.title")}</DialogTitle>
        </DialogHeader>

        <form onSubmit={onSubmit} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="organization_alias"
              label={t("organizations.createDialog.fields.name")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} />}
            </FormField>

            <FormField control={form.control} name="models" label={t("organizations.createDialog.fields.models")}>
              {(field) => (
                <ModelSelect
                  value={field.value}
                  onChange={field.onChange}
                  context="organization"
                  options={{ includeSpecialOptions: true, showAllProxyModelsOverride: true }}
                />
              )}
            </FormField>

            <FormField
              control={form.control}
              name="max_budget"
              label={t("organizations.createDialog.fields.maxBudget")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step="any" min={0} />}
            </FormField>

            <FormField
              control={form.control}
              name="budget_duration"
              label={t("organizations.createDialog.fields.budgetDuration")}
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

            <FormField control={form.control} name="tpm_limit" label={t("organizations.createDialog.fields.tpmLimit")}>
              {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
            </FormField>

            <FormField control={form.control} name="rpm_limit" label={t("organizations.createDialog.fields.rpmLimit")}>
              {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
            </FormField>

            <FormField
              control={form.control}
              name="vector_stores"
              label={t("organizations.createDialog.fields.vectorStores")}
              description={t("organizations.createDialog.fields.vectorStoresDescription")}
            >
              {(field) => (
                <VectorStoreSelector
                  value={field.value}
                  onChange={field.onChange}
                  accessToken={accessToken}
                  placeholder={t("organizations.createDialog.fields.vectorStoresPlaceholder")}
                />
              )}
            </FormField>

            <FormField
              control={form.control}
              name="mcp"
              label={t("organizations.createDialog.fields.mcp")}
              description={t("organizations.createDialog.fields.mcpDescription")}
            >
              {(field) => (
                <MCPServerSelector
                  value={field.value}
                  onChange={field.onChange}
                  accessToken={accessToken}
                  placeholder={t("organizations.createDialog.fields.mcpPlaceholder")}
                />
              )}
            </FormField>

            <FormField control={form.control} name="metadata" label={t("organizations.createDialog.fields.metadata")}>
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
              {t("organizations.createDialog.cancel")}
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? t("organizations.createDialog.creating") : t("organizations.createDialog.submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
