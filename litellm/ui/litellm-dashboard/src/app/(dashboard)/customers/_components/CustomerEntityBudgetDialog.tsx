"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { $api } from "@/lib/http/api";
import { ErrorState } from "@/components/shared/ErrorState";
import { CustomerBudgetDialog, CustomerBudgetForm } from "./CustomerBudgetDialog";
import CustomerHierarchyBudgetFields from "./CustomerHierarchyBudgetFields";
import { useCustomerHierarchyForm } from "./useCustomerHierarchyForm";
import { customerHierarchyBudget, type CustomerHierarchyEntity, type Organization } from "./customerHierarchyForms";

function BudgetForm({
  entity,
  organization,
  onPendingChange,
  onClose,
}: {
  entity: CustomerHierarchyEntity;
  organization: Organization;
  onPendingChange: (pending: boolean) => void;
  onClose: () => void;
}) {
  const form = useCustomerHierarchyForm(entity, organization, onClose, onPendingChange);
  return (
    <CustomerBudgetForm
      pending={form.pending}
      error={form.error}
      dirty={form.dirty}
      onSubmit={form.save}
      onClose={onClose}
    >
      <CustomerHierarchyBudgetFields
        kind={entity.kind}
        organization={organization}
        fields={form.fields}
        onChange={form.update}
        idPrefix="edit-customer-budget"
        compact
        original={customerHierarchyBudget(entity)}
      />
    </CustomerBudgetForm>
  );
}

export default function CustomerEntityBudgetDialog({
  organizationId,
  memberId,
  onClose,
}: {
  organizationId: string;
  memberId?: string;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const organization = $api.useQuery("get", "/management/v1/customer_organizations/{customer_org_id}", {
    params: { path: { customer_org_id: organizationId } },
  });
  const member = $api.useQuery(
    "get",
    "/management/v1/customer_members/{customer_member_id}",
    {
      params: { path: { customer_member_id: memberId ?? "" } },
    },
    { enabled: Boolean(memberId) },
  );
  const entity: CustomerHierarchyEntity | undefined = memberId
    ? member.data && { kind: "member", record: member.data }
    : organization.data && { kind: "organization", record: organization.data };
  const loading = organization.isPending || (Boolean(memberId) && member.isPending);
  const error = organization.isError || (Boolean(memberId) && member.isError);
  return (
    <CustomerBudgetDialog
      title={t(memberId ? "customerPortal.budgetEditor.memberTitle" : "customerPortal.budgetEditor.title")}
      description={entity?.record.alias || memberId || organizationId}
      pending={pending}
      onClose={onClose}
    >
      {loading && <p className="p-6">{t("customerPortal.common.loading")}</p>}
      {error && (
        <ErrorState
          onRetry={() => void Promise.all([organization.refetch(), ...(memberId ? [member.refetch()] : [])])}
        />
      )}
      {entity && organization.data && (
        <BudgetForm
          key={entity.record.updated_at}
          entity={entity}
          organization={organization.data}
          onPendingChange={setPending}
          onClose={onClose}
        />
      )}
    </CustomerBudgetDialog>
  );
}
