import React, { useEffect, useId } from "react";
import { useTranslation } from "react-i18next";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetFormPayload, COMBINED_BUDGET_DURATION } from "./budgetLimits";
import { BudgetFormFields, type BudgetFormValues, useBudgetForm } from "./BudgetFormFields";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const toFormValues = (budget: budgetItem) => {
  const monthly = budget.budget_limits?.find((limit) => limit.budget_duration === "1mo");
  const annual = budget.budget_limits?.find((limit) => limit.budget_duration === "12mo");
  const combined = monthly != null && annual != null;
  const window = monthly ?? annual;
  const windowDuration = window?.budget_duration === "1mo" ? "30d" : window?.budget_duration;
  const mixed = window != null && budget.max_budget != null;
  return {
    budget_id: budget.budget_id,
    tpm_limit: budget.tpm_limit,
    rpm_limit: budget.rpm_limit,
    tpd_limit: budget.tpd_limit,
    max_budget: window?.max_budget ?? budget.max_budget,
    budget_duration: combined ? COMBINED_BUDGET_DURATION : windowDuration ?? budget.budget_duration,
    annual_budget: annual?.max_budget,
    ...(mixed
      ? { additional_max_budget: budget.max_budget, additional_budget_duration: budget.budget_duration ?? null }
      : {}),
  };
};

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

  useEffect(() => {
    if (form.getValues("budget_id") !== existingBudget.budget_id || !form.formState.isDirty) {
      form.reset(toFormValues(existingBudget));
    }
  }, [existingBudget, form]);

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset();
  };

  const handleUpdate = async (formValues: BudgetFormValues) => {
    try {
      toast.info(t("budgets.editModal.toast.pending"));
      await updateBudget.mutateAsync(budgetFormPayload(formValues, dirtyFields, existingBudget.budget_limits));
      toast.success(t("budgets.editModal.toast.updated"));
      form.reset();
      setIsModalVisible(false);
    } catch (error) {
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
        </form>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t("budgets.editModal.cancel")}</DialogClose>
          <Button type="submit" form={formId}>
            {t("budgets.editModal.submit")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditBudgetModal;
