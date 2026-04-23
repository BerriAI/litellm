import React from "react";
import { useCreateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import NotificationsManager from "../molecules/notifications_manager";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Controller, FormProvider, useForm } from "react-hook-form";

interface BudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
}

type BudgetFormValues = {
  budget_id: string;
  tpm_limit: number | null;
  rpm_limit: number | null;
  max_budget: number | null;
  budget_duration: string | null;
};

const defaultValues: BudgetFormValues = {
  budget_id: "",
  tpm_limit: null,
  rpm_limit: null,
  max_budget: null,
  budget_duration: null,
};

function LabeledRow({
  id,
  label,
  help,
  children,
  required,
  error,
}: {
  id?: string;
  label: string;
  help?: string;
  children: React.ReactNode;
  required?: boolean;
  error?: string;
}) {
  return (
    <div className="grid grid-cols-3 gap-3 items-start mb-4">
      <Label htmlFor={id} className="pt-2">
        {label}
        {required && <span className="text-destructive">&nbsp;*</span>}
      </Label>
      <div className="col-span-2 space-y-1">
        {children}
        {help && <p className="text-xs text-muted-foreground">{help}</p>}
        {error && <p className="text-sm text-destructive">{error}</p>}
      </div>
    </div>
  );
}

const BudgetModal: React.FC<BudgetModalProps> = ({
  isModalVisible,
  setIsModalVisible,
}) => {
  const form = useForm<BudgetFormValues>({ defaultValues, mode: "onSubmit" });
  const createBudget = useCreateBudget();

  const handleCancel = () => {
    setIsModalVisible(false);
    form.reset(defaultValues);
  };

  const handleCreate = form.handleSubmit(async (values) => {
    try {
      NotificationsManager.info("Making API Call");
      await createBudget.mutateAsync(values);
      NotificationsManager.success("Budget Created");
      form.reset(defaultValues);
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error creating the budget:", error);
      NotificationsManager.fromBackend(`Error creating the budget: ${error}`);
    }
  });

  return (
    <Dialog
      open={isModalVisible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Create Budget</DialogTitle>
          <DialogDescription className="sr-only">
            Create a new spend / rate-limit budget.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleCreate}>
            <LabeledRow
              id="budget_id"
              label="Budget ID"
              help="A human-friendly name for the budget"
              required
              error={form.formState.errors.budget_id?.message as string | undefined}
            >
              <Input
                id="budget_id"
                {...form.register("budget_id", {
                  required: "Please input a human-friendly name for the budget",
                })}
              />
            </LabeledRow>
            <LabeledRow
              id="tpm_limit"
              label="Max Tokens per minute"
              help="Default is model limit."
            >
              <Input
                id="tpm_limit"
                type="number"
                step={1}
                {...form.register("tpm_limit", { valueAsNumber: true })}
              />
            </LabeledRow>
            <LabeledRow
              id="rpm_limit"
              label="Max Requests per minute"
              help="Default is model limit."
            >
              <Input
                id="rpm_limit"
                type="number"
                step={1}
                {...form.register("rpm_limit", { valueAsNumber: true })}
              />
            </LabeledRow>

            <Accordion type="single" collapsible className="mt-8 mb-4">
              <AccordionItem value="optional">
                <AccordionTrigger className="font-semibold">
                  Optional Settings
                </AccordionTrigger>
                <AccordionContent>
                  <LabeledRow id="max_budget" label="Max Budget (USD)">
                    <Input
                      id="max_budget"
                      type="number"
                      step={0.01}
                      {...form.register("max_budget", { valueAsNumber: true })}
                    />
                  </LabeledRow>
                  <LabeledRow label="Reset Budget">
                    <Controller
                      control={form.control}
                      name="budget_duration"
                      render={({ field }) => (
                        <Select
                          value={field.value ?? ""}
                          onValueChange={(v) => field.onChange(v || null)}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="n/a" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="24h">daily</SelectItem>
                            <SelectItem value="7d">weekly</SelectItem>
                            <SelectItem value="30d">monthly</SelectItem>
                          </SelectContent>
                        </Select>
                      )}
                    />
                  </LabeledRow>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            <DialogFooter className="mt-2">
              <Button type="button" variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
              <Button type="submit" disabled={createBudget.isPending}>
                {createBudget.isPending ? "Creating…" : "Create Budget"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default BudgetModal;
