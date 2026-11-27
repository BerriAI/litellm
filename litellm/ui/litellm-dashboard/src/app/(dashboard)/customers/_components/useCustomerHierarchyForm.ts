import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { saveCustomerHierarchy } from "./customerHierarchyApi";
import { useCustomerBudgetForm } from "./useCustomerBudgetForm";
import {
  buildCustomerHierarchyPatch,
  customerEntityId,
  customerHierarchyFields,
  CUSTOMER_HIERARCHY_FORM_ERRORS,
  type CustomerHierarchyEntity,
  type CustomerHierarchyPatch,
  type Organization,
} from "./customerHierarchyForms";

export function useSaveCustomerHierarchy(entity: CustomerHierarchyEntity, onSaved: () => void) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  return async (patch: CustomerHierarchyPatch, onMissing: () => void): Promise<void> => {
    const saved = await saveCustomerHierarchy(customerEntityId(entity), patch);
    if (!saved.data) return onMissing();
    await queryClient.cancelQueries({ queryKey: saved.queryKey, exact: true });
    queryClient.setQueryData(saved.queryKey, saved.data);
    void queryClient.invalidateQueries({ queryKey: ["customer-hierarchy"] });
    void queryClient.invalidateQueries({ queryKey: ["customer-organizations"] });
    void queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_teams/{customer_team_id}"] });
    void queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_members/{customer_member_id}"] });
    void queryClient.invalidateQueries({
      queryKey: ["get", "/management/v1/customer_organizations/{customer_org_id}"],
    });
    toast.success(t("customerPortal.hierarchy.saved"));
    onSaved();
  };
}

export function useCustomerHierarchyForm(
  entity: CustomerHierarchyEntity,
  organization: Organization,
  onSaved: () => void,
  onPendingChange?: (pending: boolean) => void,
) {
  const { t } = useTranslation();
  const write = useSaveCustomerHierarchy(entity, onSaved);
  const form = useCustomerBudgetForm(
    {
      initial: customerHierarchyFields(entity, organization),
      parse: (fields) => {
        const result = buildCustomerHierarchyPatch(entity, fields, organization);
        return result.ok
          ? { ok: true, value: result.patch }
          : { ok: false, error: t(CUSTOMER_HIERARCHY_FORM_ERRORS[result.error]) };
      },
      write: (patch) => write(patch, () => form.setError(t("customerPortal.hierarchy.error"))),
    },
    onPendingChange,
  );
  return form;
}
