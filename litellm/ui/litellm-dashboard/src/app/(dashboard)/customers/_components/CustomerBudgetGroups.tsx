"use client";

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import type { Organization } from "./customerHierarchyForms";
import CustomerBudgetGroupsEditor from "./CustomerBudgetGroupsEditor";
import { customerBudgetGroupRows, planCustomerBudgetGroups } from "./customerBudgetGroupsPlan";
import type { CustomerBudgetGroupFields, CustomerOrganizationFormError } from "./customerOrganizationForm";

export default function CustomerBudgetGroups({
  organization,
  onSaved,
}: {
  organization: Organization;
  onSaved: () => void;
}) {
  const { t } = useTranslation();
  const [groups, setGroups] = useState(() => customerBudgetGroupRows(organization));
  const [defaultGroupId, setDefaultGroupId] = useState(organization.default_budget_template_id ?? "");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<CustomerOrganizationFormError | "renameCollision" | null>(null);
  const plan = planCustomerBudgetGroups(organization, groups, defaultGroupId);
  const unchanged =
    JSON.stringify(groups) === JSON.stringify(customerBudgetGroupRows(organization)) &&
    defaultGroupId === (organization.default_budget_template_id ?? "");
  const changeGroups = (next: CustomerBudgetGroupFields[], defaultId: string) => {
    setGroups(next);
    setDefaultGroupId(defaultId);
    setError(null);
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (pending) return;
    if (!plan.ok) {
      setError(plan.error);
      return;
    }
    setPending(true);
    const path = { customer_org_id: organization.customer_org_id };
    try {
      const created = [];
      for (const item of plan.creates) {
        const response = await fetchClient.POST(
          "/management/v1/customer_organizations/{customer_org_id}/default_budgets",
          {
            params: { path },
            body: item.body,
          },
        );
        const template = response.data?.default_budgets.find((group) => group.name === item.body.name);
        if (!template) {
          toast.error(t("customerPortal.budgetGroups.missingCreated"));
          return;
        }
        created.push({ rowId: item.rowId, budgetId: template.budget_id });
      }
      for (const item of plan.updates) {
        await fetchClient.PATCH("/management/v1/customer_organizations/{customer_org_id}/default_budgets/{budget_id}", {
          params: { path: { ...path, budget_id: item.budgetId } },
          body: item.body,
        });
      }
      if (plan.defaultChange) {
        const budgetId =
          created.find((item) => item.rowId === plan.defaultChange?.rowId)?.budgetId ?? plan.defaultChange.rowId;
        await fetchClient.PATCH("/management/v1/customer_organizations/{customer_org_id}", {
          params: { path },
          body: { default_budget_template_id: budgetId },
        });
      }
      for (const budgetId of plan.removes) {
        await fetchClient.DELETE(
          "/management/v1/customer_organizations/{customer_org_id}/default_budgets/{budget_id}",
          {
            params: { path: { ...path, budget_id: budgetId } },
          },
        );
      }
      toast.success(t("customerPortal.hierarchy.saved"));
    } catch (failure) {
      toast.error(extractProxyErrorMessage(failure));
    } finally {
      onSaved();
      setPending(false);
    }
  };
  return (
    <form onSubmit={save} className="max-w-3xl space-y-5">
      <fieldset disabled={pending}>
        <CustomerBudgetGroupsEditor
          groups={groups}
          defaultGroupId={defaultGroupId}
          disabled={pending}
          onChange={changeGroups}
        />
      </fieldset>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error === "renameCollision"
            ? t("customerPortal.budgetGroups.renameCollision")
            : t(`customerPortal.createDialog.errors.${error}`)}
        </p>
      )}
      <Button type="submit" disabled={pending || unchanged || (plan.ok && !plan.changed)}>
        {t("customerPortal.hierarchy.save")}
      </Button>
    </form>
  );
}
