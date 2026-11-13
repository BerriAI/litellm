import React, { useEffect, useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetProfilePayload } from "./budgetProfilePayload";
import { BudgetFormFields, type BudgetFormValues, useBudgetForm } from "./BudgetFormFields";
import { toast } from "@/lib/toast";
import { ApiError } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const toFormValues = (budget: budgetItem) => ({
  budget_id: budget.budget_id,
  tpm_limit: budget.tpm_limit,
  rpm_limit: budget.rpm_limit,
  tpd_limit: budget.tpd_limit,
  max_budget: budget.max_budget,
  budget_duration: budget.budget_duration,
  is_profile: budget.is_profile,
  revision: budget.revision,
  monthly: budget.budget_limits?.find((limit) => limit.budget_duration === "1mo")?.max_budget ?? null,
  annual: budget.budget_limits?.find((limit) => limit.budget_duration === "12mo")?.max_budget ?? null,
  cap_policy: budget.cap_policy ?? "pause_until_reset",
});

interface EditBudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  existingBudget: budgetItem;
}
const EditBudgetModal: React.FC<EditBudgetModalProps> = ({ isModalVisible, setIsModalVisible, existingBudget }) => {
  const { t } = useTranslation();
  const formId = useId();
  const form = useBudgetForm(toFormValues(existingBudget));
  const { dirtyFields } = form.formState;
  const updateBudget = useUpdateBudget();
  const [conflict, setConflict] = useState(false);

  useEffect(() => {
    if (form.getValues("budget_id") !== existingBudget.budget_id || !form.formState.isDirty) {
      form.reset(toFormValues(existingBudget));
    }
  }, [existingBudget, form]);

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset();
    setConflict(false);
  };

  const handleUpdate = async (formValues: BudgetFormValues) => {
    try {
      toast.info(t("budgets.editModal.toast.pending"));
      await updateBudget.mutateAsync(budgetProfilePayload(formValues, dirtyFields));
      toast.success(t("budgets.editModal.toast.updated"));
      form.reset();
      setIsModalVisible(false);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setConflict(true);
        return;
      }
      console.error("Error updating the budget:", error);
      toast.fromError(t("budgets.editModal.toast.error", { error: String(error) }));
    }
  };

  return (
    <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-[15px] leading-[22px] font-semibold">{t("budgets.editModal.title")}</DialogTitle>
        </DialogHeader>
        <form id={formId} onSubmit={form.handleSubmit(handleUpdate)} noValidate>
          <BudgetFormFields form={form} isEditing />
          {conflict && (
            <p role="alert" className="mt-3 text-sm text-destructive">
              {t("budgetProfiles.profileConflict")}
            </p>
          )}
        </form>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t("budgets.editModal.cancel")}</DialogClose>
          <Button type="submit" form={formId} disabled={updateBudget.isPending || conflict}>
            {t("budgets.editModal.submit")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditBudgetModal;
