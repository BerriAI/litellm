"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { CustomerBudgetDialog, CustomerBudgetForm } from "./CustomerBudgetDialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ErrorState } from "@/components/shared/ErrorState";
import {
  buildCustomerOrganizationBody,
  initialCustomerOrganizationFields,
  type CustomerOrganizationFormError,
} from "./customerOrganizationForm";

import CustomerLimitsFields, { type CustomerLimits } from "./CustomerLimitsFields";
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
  const [fields, setFields] = useState(initialCustomerOrganizationFields);
  const [error, setError] = useState<CustomerOrganizationFormError | null>(null);
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {}, { enabled: open });
  const defaultBudgetId = groups.data?.default_budget_template_id;
  const effectiveFields =
    fields.budgetMode === "inherit" && groups.isSuccess
      ? {
          ...fields,
          budgetMode: defaultBudgetId ? ("profile" as const) : ("custom" as const),
          budgetId: defaultBudgetId ?? "",
        }
      : fields;
  const create = $api.useMutation("post", "/management/v1/customer_organizations", {
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_portal/limits"] }),
    onSuccess: (customer) => {
      toast.success(t("customerPortal.toasts.created", { name: customer.alias || fields.name.trim() }));
      onCreated(customer.customer_org_id);
      setFields(initialCustomerOrganizationFields());
      setError(null);
      onOpenChange(false);
    },
    onError: (e: unknown) =>
      toast.error(t("customerPortal.toasts.createFailed", { error: extractProxyErrorMessage(e) })),
  });
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (create.isPending || groups.isPending) return;
    const result = buildCustomerOrganizationBody(fields);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setError(null);
    create.mutate({ body: result.body });
  };
  const changeLimits = (patch: Partial<CustomerLimits>) => {
    setFields({ ...effectiveFields, ...patch });
    setError(null);
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
        dirty={Boolean(fields.name.trim()) && !groups.isPending}
        saveLabel={t(create.isPending ? "customerPortal.createDialog.creating" : "customerPortal.createDialog.submit")}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="cc-name" className="font-normal text-muted-foreground">
              {t("customerPortal.createDialog.name")}
            </Label>
            <Input
              id="cc-name"
              value={fields.name}
              onChange={(event) => setFields({ ...fields, name: event.target.value })}
              placeholder={t("customerPortal.createDialog.namePlaceholder")}
              autoFocus
              required
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cc-org-id" className="font-normal text-muted-foreground">
              {t("customerPortal.createDialog.customerIdOptional")}
            </Label>
            <Input
              id="cc-org-id"
              value={fields.customerOrgId}
              onChange={(event) => setFields({ ...fields, customerOrgId: event.target.value })}
              placeholder={t("customerPortal.createDialog.customerIdPlaceholder")}
            />
          </div>
        </div>
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
          onChange={changeLimits}
        />
        {groups.isError && <ErrorState onRetry={() => void groups.refetch()} />}
        <CustomerBudgetGroupsEditor
          groups={fields.groups}
          defaultGroupId={fields.defaultGroupId}
          disabled={create.isPending}
          onChange={(groups, defaultGroupId) => {
            setFields({ ...fields, groups, defaultGroupId });
            setError(null);
          }}
        />
      </CustomerBudgetForm>
    </CustomerBudgetDialog>
  );
}
