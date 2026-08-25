import { ChevronRight } from "lucide-react";
import React, { useEffect } from "react";
import { useForm } from "react-hook-form";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { applyBudgetPrecision } from "./budgetPrecision";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

type EditBudgetFormValues = Pick<
  budgetItem,
  "budget_id" | "tpm_limit" | "rpm_limit" | "max_budget" | "budget_duration"
>;

const toFormValues = (budget: budgetItem): EditBudgetFormValues => ({
  budget_id: budget.budget_id,
  tpm_limit: budget.tpm_limit,
  rpm_limit: budget.rpm_limit,
  max_budget: budget.max_budget,
  budget_duration: budget.budget_duration,
});

const BUDGET_DURATION_OPTIONS = [
  { value: "24h", label: "daily" },
  { value: "7d", label: "weekly" },
  { value: "30d", label: "monthly" },
];

interface EditBudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  existingBudget: budgetItem;
}
const EditBudgetModal: React.FC<EditBudgetModalProps> = ({ isModalVisible, setIsModalVisible, existingBudget }) => {
  const [optionalSettingsOpen, setOptionalSettingsOpen] = React.useState(false);
  const form = useForm<EditBudgetFormValues>({ defaultValues: toFormValues(existingBudget) });
  const updateBudget = useUpdateBudget();

  useEffect(() => {
    form.reset(toFormValues(existingBudget));
  }, [existingBudget, form]);

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset();
  };

  const handleUpdate = async (formValues: EditBudgetFormValues) => {
    try {
      toast.info("Making API Call");
      await updateBudget.mutateAsync(
        applyBudgetPrecision(
          optionalSettingsOpen ? formValues : { ...formValues, max_budget: undefined, budget_duration: undefined },
        ),
      );
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
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Edit Budget</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleUpdate)} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="budget_id"
              label="Budget ID"
              description="Budget ID cannot be changed after creation"
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} disabled />}
            </FormField>
            <FormField
              control={form.control}
              name="tpm_limit"
              label="Max Tokens per minute"
              description="Default is model limit."
            >
              {({ ref, value, onChange, ...field }) => (
                <Input
                  {...field}
                  ref={ref}
                  type="number"
                  step={1}
                  value={value ?? ""}
                  onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
                />
              )}
            </FormField>
            <FormField
              control={form.control}
              name="rpm_limit"
              label="Max Requests per minute"
              description="Default is model limit."
            >
              {({ ref, value, onChange, ...field }) => (
                <Input
                  {...field}
                  ref={ref}
                  type="number"
                  step={1}
                  value={value ?? ""}
                  onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
                />
              )}
            </FormField>

            <Collapsible open={optionalSettingsOpen} onOpenChange={setOptionalSettingsOpen} className="mt-20 mb-8">
              <CollapsibleTrigger className="group flex w-full items-center justify-between py-2 text-left">
                <b>Optional Settings</b>
                <ChevronRight className="size-4 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
              </CollapsibleTrigger>
              <CollapsibleContent>
                <FormField control={form.control} name="max_budget" label="Max Budget (USD)">
                  {({ ref, value, onChange, ...field }) => (
                    <Input
                      {...field}
                      ref={ref}
                      type="number"
                      step={0.01}
                      value={value ?? ""}
                      onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
                    />
                  )}
                </FormField>
                <FormField className="mt-8" control={form.control} name="budget_duration" label="Reset Budget">
                  {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                    <Select items={BUDGET_DURATION_OPTIONS} value={value ?? null} onValueChange={onChange}>
                      <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                        <SelectValue placeholder="n/a" />
                      </SelectTrigger>
                      <SelectContent>
                        {BUDGET_DURATION_OPTIONS.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </FormField>
              </CollapsibleContent>
            </Collapsible>
          </FieldGroup>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button type="submit">Save</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default EditBudgetModal;
