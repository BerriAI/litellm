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
import type { CustomerBudgetGroupFields } from "./customerOrganizationForm";

interface Draft {
  base: Organization;
  groups: CustomerBudgetGroupFields[];
  defaultGroupId: string;
}

export default function CustomerBudgetGroups({
  organization,
  onSaved,
}: {
  organization: Organization;
  onSaved: (organization: Organization) => void | Promise<void>;
}) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const base = draft?.base ?? organization;
  const groups = draft?.groups ?? customerBudgetGroupRows(base);
  const defaultGroupId = draft?.defaultGroupId ?? base.default_budget_template_id ?? "";
  const plan = planCustomerBudgetGroups(base, groups, defaultGroupId);
  const unchanged = plan.ok && !plan.changed;
  const changeGroups = (groups: CustomerBudgetGroupFields[], defaultGroupId: string) => {
    setDraft({ base, groups, defaultGroupId });
    setError(null);
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (pending) return;
    if (!plan.ok) {
      setError(t(`customerPortal.createDialog.errors.${plan.error}`));
      return;
    }
    setPending(true);
    setError(null);
    try {
      const response = await fetchClient.PATCH(
        "/management/v1/customer_organizations/{customer_org_id}/default_budgets",
        { params: { path: { customer_org_id: base.customer_org_id } }, body: plan.body },
      );
      if (!response.data) throw new Error(t("customerPortal.hierarchy.error"));
      await onSaved(response.data);
      setDraft(null);
      toast.success(t("customerPortal.hierarchy.saved"));
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  const reload = async () => {
    setPending(true);
    try {
      const response = await fetchClient.GET("/management/v1/customer_organizations/{customer_org_id}", {
        params: { path: { customer_org_id: base.customer_org_id } },
      });
      if (!response.data) throw new Error(t("customerPortal.hierarchy.error"));
      await onSaved(response.data);
      setDraft(null);
      setError(null);
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
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
          {error}
        </p>
      )}
      <div className="flex gap-3">
        <Button type="submit" disabled={pending || !draft || unchanged}>
          {t("customerPortal.hierarchy.save")}
        </Button>
        {draft && (
          <Button type="button" variant="outline" disabled={pending} onClick={() => void reload()}>
            {t("customerPortal.budgetGroups.discard")}
          </Button>
        )}
      </div>
    </form>
  );
}
