import React, { useEffect, useId } from "react";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { applyBudgetPrecision } from "./budgetPrecision";
import { BudgetFormFields, type BudgetFormValues, useBudgetForm } from "./BudgetFormFields";
import { pickDirty } from "@/lib/forms/pickDirty";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const toFormValues = (budget: budgetItem) => ({
  budget_id: budget.budget_id,
  tpm_limit: budget.tpm_limit,
  rpm_limit: budget.rpm_limit,
  tpd_limit: budget.tpd_limit,
  max_budget: budget.max_budget,
  budget_duration: budget.budget_duration,
});

interface EditBudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  existingBudget: budgetItem;
}
const EditBudgetModal: React.FC<EditBudgetModalProps> = ({ isModalVisible, setIsModalVisible, existingBudget }) => {
  const formId = useId();
  const form = useBudgetForm(toFormValues(existingBudget));
  const { dirtyFields } = form.formState;
  const updateBudget = useUpdateBudget();

  useEffect(() => {
    form.reset(toFormValues(existingBudget));
  }, [existingBudget, form]);

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset();
  };

  const handleUpdate = async (formValues: BudgetFormValues) => {
    try {
      toast.info("Making API Call");
      const changedValues = pickDirty(formValues, dirtyFields);
      await updateBudget.mutateAsync(applyBudgetPrecision({ budget_id: formValues.budget_id, ...changedValues }));
      toast.success("Budget Updated");
      form.reset();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error updating the budget:", error);
      toast.fromError(`Error updating the budget: ${error}`);
    }
  };

  return (
    <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-[15px] leading-[22px] font-semibold">Edit budget</DialogTitle>
        </DialogHeader>
        <form id={formId} onSubmit={form.handleSubmit(handleUpdate)} noValidate>
          <BudgetFormFields form={form} isEditing />
        </form>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>Cancel</DialogClose>
          <Button type="submit" form={formId}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditBudgetModal;
