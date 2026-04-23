import React, { useEffect } from "react";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "./budget_panel";
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

interface EditBudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  existingBudget: budgetItem;
}

type EditBudgetFormValues = {
  budget_id: string;
  tpm_limit: number | null;
  rpm_limit: number | null;
  max_budget: number | null;
  budget_duration: string | null;
};

function LabeledRow({
  id,
  label,
  help,
  children,
  disabledHint,
}: {
  id?: string;
  label: string;
  help?: string;
  children: React.ReactNode;
  disabledHint?: string;
}) {
  return (
    <div className="grid grid-cols-3 gap-3 items-start mb-4">
      <Label htmlFor={id} className="pt-2">
        {label}
      </Label>
      <div className="col-span-2 space-y-1">
        {children}
        {help && <p className="text-xs text-muted-foreground">{help}</p>}
        {disabledHint && (
          <p className="text-xs text-muted-foreground">{disabledHint}</p>
        )}
      </div>
    </div>
  );
}

const EditBudgetModal: React.FC<EditBudgetModalProps> = ({
  isModalVisible,
  setIsModalVisible,
  existingBudget,
}) => {
  const form = useForm<EditBudgetFormValues>({
    defaultValues: {
      budget_id: existingBudget.budget_id,
      tpm_limit: existingBudget.tpm_limit,
      rpm_limit: existingBudget.rpm_limit,
      max_budget: existingBudget.max_budget,
      budget_duration: null,
    },
    mode: "onSubmit",
  });
  const updateBudget = useUpdateBudget();

  useEffect(() => {
    form.reset({
      budget_id: existingBudget.budget_id,
      tpm_limit: existingBudget.tpm_limit,
      rpm_limit: existingBudget.rpm_limit,
      max_budget: existingBudget.max_budget,
      budget_duration: null,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [existingBudget]);

  const handleCancel = () => {
    setIsModalVisible(false);
  };

  const handleUpdate = form.handleSubmit(async (values) => {
    try {
      NotificationsManager.info("Making API Call");
      await updateBudget.mutateAsync(values);
      NotificationsManager.success("Budget Updated");
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error updating the budget:", error);
      NotificationsManager.fromBackend(`Error updating the budget: ${error}`);
    }
  });

  return (
    <Dialog
      open={isModalVisible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Edit Budget</DialogTitle>
          <DialogDescription className="sr-only">
            Update an existing budget. Budget ID cannot be changed after creation.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleUpdate}>
            <LabeledRow
              id="budget_id"
              label="Budget ID"
              help="Budget ID cannot be changed after creation"
            >
              <Input id="budget_id" {...form.register("budget_id")} disabled />
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
              <Button type="submit" disabled={updateBudget.isPending}>
                {updateBudget.isPending ? "Saving…" : "Save"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default EditBudgetModal;
