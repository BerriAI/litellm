import { ChevronRight } from "lucide-react";
import React from "react";
import { z } from "zod/v4";
import { useCreateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { applyBudgetPrecision } from "./budgetPrecision";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const budgetShape = {
  budget_id: z.string().min(1, "Please input a human-friendly name for the budget"),
  tpm_limit: z.number().nullish(),
  rpm_limit: z.number().nullish(),
  max_budget: z.number().nullish(),
  budget_duration: z.string().nullish(),
};

const budgetSchema = z.object(budgetShape);

type BudgetFormValues = z.output<typeof budgetSchema>;

const BUDGET_DURATION_OPTIONS = [
  { value: "24h", label: "daily" },
  { value: "7d", label: "weekly" },
  { value: "30d", label: "monthly" },
];

interface BudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
}
const BudgetModal: React.FC<BudgetModalProps> = ({ isModalVisible, setIsModalVisible }) => {
  const [optionalSettingsOpen, setOptionalSettingsOpen] = React.useState(false);
  const form = useZodForm(budgetSchema, { defaultValues: { budget_id: "" } });
  const createBudget = useCreateBudget();

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset();
  };

  const handleCreate = async (formValues: BudgetFormValues) => {
    try {
      toast.info("Making API Call");
      await createBudget.mutateAsync(
        applyBudgetPrecision(
          optionalSettingsOpen ? formValues : { ...formValues, max_budget: undefined, budget_duration: undefined },
        ),
      );
      toast.success("Budget Created");
      form.reset();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error creating the budget:", error);
      toast.fromError(`Error creating the budget: ${error}`);
    }
  };

  return (
    <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Create Budget</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleCreate)} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="budget_id"
              label="Budget ID"
              description="A human-friendly name for the budget"
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} placeholder="" />}
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
            <Button type="submit">Create Budget</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default BudgetModal;
