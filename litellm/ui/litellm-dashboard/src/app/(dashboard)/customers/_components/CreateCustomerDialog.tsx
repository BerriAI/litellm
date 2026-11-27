"use client";

import { useState } from "react";
import { Controller } from "react-hook-form";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useZodForm } from "@/lib/forms/useZodForm";
import { CustomerBudgetDialog, CustomerBudgetForm } from "./CustomerBudgetDialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ErrorState } from "@/components/shared/ErrorState";
import type { CustomerOrganizationFormError } from "./customerOrganizationForm";
import { customerCreateDefaults, toCustomerCreateBody } from "./customer-create/mapper";
import { customerCreateSchema } from "./customer-create/schema";
import CustomerLimitsFields from "./CustomerLimitsFields";
import CustomerBudgetGroupsEditor from "./CustomerBudgetGroupsEditor";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (customerOrgId: string) => void;
}

const limitsIdPrefix = "cc";

export default function CreateCustomerDialog({ open, onOpenChange, onCreated }: Props) {
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const { control, watch, getValues, reset } = useZodForm(customerCreateSchema, {
    defaultValues: customerCreateDefaults(),
  });
  const name = watch("name");
  const [error, setError] = useState<CustomerOrganizationFormError | null>(null);
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {}, { enabled: open });
  const defaultBudgetId = groups.data?.default_budget_template_id;
  const defaultBudgetMode = defaultBudgetId ? ("profile" as const) : ("custom" as const);
  const create = $api.useMutation("post", "/management/v1/customer_organizations", {
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_portal/limits"] }),
    onSuccess: (customer) => {
      toast.success(t("customerPortal.toasts.created", { name: customer.alias || name.trim() }));
      onCreated(customer.customer_org_id);
      reset(customerCreateDefaults());
      setError(null);
      onOpenChange(false);
    },
    onError: (e: unknown) =>
      toast.error(t("customerPortal.toasts.createFailed", { error: extractProxyErrorMessage(e) })),
  });
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (create.isPending || groups.isPending) return;
    const fields = getValues();
    const parsed = customerCreateSchema.safeParse(fields);
    const result = parsed.success ? parsed.data : toCustomerCreateBody(fields);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setError(null);
    create.mutate({ body: result.body });
  };
  return (
    <CustomerBudgetDialog
      open={open}
      onClose={() => onOpenChange(false)}
      pending={create.isPending}
      title={t("customerPortal.createDialog.title")}
      className="max-h-[90dvh] w-[calc(100%-2rem)] sm:max-w-5xl"
    >
      <CustomerBudgetForm
        onSubmit={submit}
        pending={create.isPending}
        error={error ? t(`customerPortal.createDialog.errors.${error}`) : null}
        onClose={() => onOpenChange(false)}
        dirty={Boolean(name.trim()) && !groups.isPending}
        saveLabel={t(create.isPending ? "customerPortal.createDialog.creating" : "customerPortal.createDialog.submit")}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="cc-name" className="font-normal text-muted-foreground">
              {t("customerPortal.createDialog.name")}
            </Label>
            <Controller
              control={control}
              name="name"
              render={({ field }) => (
                <Input
                  id="cc-name"
                  ref={field.ref}
                  value={field.value}
                  onChange={field.onChange}
                  placeholder={t("customerPortal.createDialog.namePlaceholder")}
                  autoFocus
                  required
                />
              )}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cc-org-id" className="font-normal text-muted-foreground">
              {t("customerPortal.createDialog.customerIdOptional")}
            </Label>
            <Controller
              control={control}
              name="customerOrgId"
              render={({ field }) => (
                <Input
                  id="cc-org-id"
                  ref={field.ref}
                  value={field.value}
                  onChange={field.onChange}
                  placeholder={t("customerPortal.createDialog.customerIdPlaceholder")}
                />
              )}
            />
          </div>
        </div>
        <Controller
          control={control}
          name="limits"
          render={({ field }) => {
            const effectiveFields =
              field.value.budgetMode === "inherit" && groups.isSuccess
                ? {
                    ...field.value,
                    budgetMode: defaultBudgetMode,
                    budgetId: defaultBudgetId ?? "",
                  }
                : field.value;
            return (
              <CustomerLimitsFields
                fields={effectiveFields}
                profiles={groups.data?.default_budgets ?? []}
                title={t("customerPortal.createDialog.organizationLimits")}
                description={t("customerPortal.createDialog.organizationLimitsHelp")}
                idPrefix={limitsIdPrefix}
                loading={groups.isPending}
                inheritedOption={{
                  label: t("customerPortal.budgetGroups.organizationDefault"),
                  summary: t("customerPortal.budgetGroups.organizationDefaultHelp"),
                }}
                disabled={create.isPending}
                onChange={(patch) => {
                  field.onChange({ ...effectiveFields, ...patch });
                  setError(null);
                }}
              />
            );
          }}
        />
        {groups.isError && <ErrorState onRetry={() => void groups.refetch()} />}
        <Controller
          control={control}
          name="budgetGroups"
          render={({ field }) => (
            <CustomerBudgetGroupsEditor
              groups={field.value.groups}
              defaultGroupId={field.value.defaultGroupId}
              disabled={create.isPending}
              onChange={(groups, defaultGroupId) => {
                field.onChange({ groups, defaultGroupId });
                setError(null);
              }}
            />
          )}
        />
      </CustomerBudgetForm>
    </CustomerBudgetDialog>
  );
}
