import { useLanguage } from "@/contexts/LanguageContext";
import React, { useId } from "react";
import { useCreateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { applyBudgetPrecision } from "./budgetPrecision";
import { BudgetFormFields, type BudgetFormValues, useBudgetForm } from "./BudgetFormFields";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import TranslatedMessage from "@/components/shared/TranslatedMessage";

interface BudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
}
const BudgetModal: React.FC<BudgetModalProps> = ({ isModalVisible, setIsModalVisible }) => {
  const { t } = useLanguage();
  const formId = useId();
  const form = useBudgetForm({ budget_id: "" });
  const createBudget = useCreateBudget();

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset();
  };

  const handleCreate = async (formValues: BudgetFormValues) => {
    try {
      toast.info(<TranslatedMessage message="Making API Call" />);
      await createBudget.mutateAsync(applyBudgetPrecision(formValues));
      toast.success(<TranslatedMessage message="Budget Created" />);
      form.reset();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error creating the budget:", error);
      // 原始错误用于状态分类，只有缺少详情时才显示可切换语言的本地提示。
      const hasEmptyMessage = error instanceof Error && !error.message;
      toast.fromError(
        error,
        error == null || error === "" || hasEmptyMessage
          ? { description: <TranslatedMessage message="Failed to create budget" /> }
          : undefined,
      );
    }
  };

  return (
    <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-[15px] leading-[22px] font-semibold">{t("Create budget")}</DialogTitle>
        </DialogHeader>
        <form id={formId} onSubmit={form.handleSubmit(handleCreate)} noValidate>
          <BudgetFormFields form={form} isEditing={false} />
        </form>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t("Cancel")}</DialogClose>
          <Button type="submit" form={formId}>
            {t("Create budget")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default BudgetModal;
