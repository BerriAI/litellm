"use client";

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import type { Organization } from "./customerHierarchyForms";
import CustomerBudgetGroupsEditor from "./CustomerBudgetGroupsEditor";
import { customerBudgetGroupRows, planCustomerBudgetGroups, type BudgetGroups } from "./customerBudgetGroupsPlan";
import type { CustomerBudgetGroupFields } from "./customerOrganizationForm";

interface Draft {
  base: BudgetGroups;
  groups: CustomerBudgetGroupFields[];
  defaultGroupId: string;
}

type Props =
  | { organization: Organization; onSaved: (organization: Organization) => void | Promise<void>; global?: false }
  | { organization: BudgetGroups; onSaved: (groups: BudgetGroups) => void | Promise<void>; global: true };

export default function CustomerBudgetGroups(props: Props) {
  const { organization } = props;
  const onSaved = async (data: BudgetGroups | Organization | undefined) => {
    if (!data) throw new Error(t("customerPortal.hierarchy.error"));
    if (props.global) await props.onSaved(data);
    else if ("customer_org_id" in data) await props.onSaved(data);
  };
  const { t } = useTranslation();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const base = draft?.base ?? organization;
  const groups = draft?.groups ?? customerBudgetGroupRows(base);
  const defaultGroupId = draft?.defaultGroupId ?? base.default_budget_template_id ?? "";
  const plan = planCustomerBudgetGroups(base, groups, defaultGroupId, !props.global);
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
      const response = props.global
        ? await fetchClient.PATCH("/management/v1/customer_organization_budget_groups", { body: plan.body })
        : await fetchClient.PATCH("/management/v1/customer_organizations/{customer_org_id}/default_budgets", {
            params: { path: { customer_org_id: props.global ? "" : props.organization.customer_org_id } },
            body: plan.body,
          });
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
      const response = props.global
        ? await fetchClient.GET("/management/v1/customer_organization_budget_groups")
        : await fetchClient.GET("/management/v1/customer_organizations/{customer_org_id}", {
            params: { path: { customer_org_id: props.global ? "" : props.organization.customer_org_id } },
          });
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
          memberGroups={!props.global}
          groups={groups}
          showHeading={false}
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
      <div className="flex justify-end gap-2 border-t pt-4">
        {draft && (
          <Button type="button" variant="outline" disabled={pending} onClick={() => void reload()}>
            {t("customerPortal.budgetGroups.discard")}
          </Button>
        )}
        <Button type="submit" disabled={pending || !draft || unchanged}>
          {t("customerPortal.hierarchy.save")}
        </Button>
      </div>
    </form>
  );
}
