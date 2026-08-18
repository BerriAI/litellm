import React from "react";
import { Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { Modal } from "antd";
import { z } from "zod/v4";
import { useCreateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { applyBudgetPrecision } from "./budgetPrecision";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useZodForm } from "@/lib/forms/useZodForm";

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
  const form = useZodForm(budgetSchema, { shouldUnregister: true, defaultValues: { budget_id: "" } });
  const createBudget = useCreateBudget();

  const handleOk = () => {
    setIsModalVisible(false);
    form.reset();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset();
  };

  const handleCreate = async (formValues: BudgetFormValues) => {
    try {
      NotificationsManager.info("Making API Call");
      await createBudget.mutateAsync(applyBudgetPrecision(formValues));
      NotificationsManager.success("Budget Created");
      form.reset();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error creating the budget:", error);
      NotificationsManager.fromBackend(`Error creating the budget: ${error}`);
    }
  };

  return (
    <Modal
      title="Create Budget"
      open={isModalVisible}
      width={800}
      footer={null}
      onOk={handleOk}
      onCancel={handleCancel}
    >
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

          <Accordion className="mt-20 mb-8">
            <AccordionHeader>
              <b>Optional Settings</b>
            </AccordionHeader>
            <AccordionBody>
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
            </AccordionBody>
          </Accordion>
        </FieldGroup>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button type="submit">Create Budget</Button>
        </div>
      </form>
    </Modal>
  );
};

export default BudgetModal;
