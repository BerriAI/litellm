"use client";

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { fetchClient } from "@/lib/http/api";
import { useCustomerBudgetSave } from "./useCustomerBudgetSave";
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
  | {
      organization: BudgetGroups;
      onSaved: (groups: BudgetGroups) => void | Promise<void>;
      global: true;
      scope?: "organization" | "app";
    };

export default function CustomerBudgetGroups(props: Props) {
  const { organization } = props;
  const globalEndpoint =
    props.global && props.scope === "app"
      ? "/management/v1/app_customer_organization_budget_groups"
      : "/management/v1/customer_organization_budget_groups";
  const onSaved = async (data: BudgetGroups | Organization | undefined) => {
    if (!data) {
      setError(t("customerPortal.hierarchy.error"));
      return false;
    }
    if (props.global) await props.onSaved(data);
    else if ("customer_org_id" in data) await props.onSaved(data);
    return true;
  };
  const { t } = useTranslation();
  const [draft, setDraft] = useState<Draft | null>(null);
  const { pending, error, setError, run } = useCustomerBudgetSave();
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
    await run(async () => {
      const response = props.global
        ? await fetchClient.PATCH(globalEndpoint, { body: plan.body })
        : await fetchClient.PATCH("/management/v1/customer_organizations/{customer_org_id}/default_budgets", {
            params: { path: { customer_org_id: props.global ? "" : props.organization.customer_org_id } },
            body: plan.body,
          });
      if (!(await onSaved(response.data))) return;
      setDraft(null);
      toast.success(t("customerPortal.hierarchy.saved"));
    });
  };
  const reload = async () => {
    await run(async () => {
      const response = props.global
        ? await fetchClient.GET(globalEndpoint)
        : await fetchClient.GET("/management/v1/customer_organizations/{customer_org_id}", {
            params: { path: { customer_org_id: props.global ? "" : props.organization.customer_org_id } },
          });
      if (!(await onSaved(response.data))) return;
      setDraft(null);
      setError(null);
    });
  };
  return (
    <form onSubmit={save} className="w-full space-y-5">
      <fieldset disabled={pending} className="min-w-0">
        <CustomerBudgetGroupsEditor
          memberGroups={!props.global}
          appGroups={props.global && props.scope === "app"}
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
