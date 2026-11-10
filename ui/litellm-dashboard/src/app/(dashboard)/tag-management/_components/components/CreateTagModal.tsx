"use client";

import { ChevronRight, CircleHelp } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import type { FieldErrors } from "react-hook-form";
import { z } from "zod";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import NumericalInput from "@/components/shared/numerical_input";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import i18n from "@/i18n/i18n";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

interface ModelInfo {
  model_name: string;
  litellm_params: {
    model: string;
  };
  model_info: {
    id: string;
  };
}

const optionalLimit = z
  .string()
  .refine((value) => value === "" || (Number.isInteger(Number(value)) && Number(value) >= 0), {
    error: () => i18n.t("tagManagement.createModal.validation.limitNonnegativeInteger"),
  })
  .optional();

const createTagShape = {
  tag_name: z.string().min(1, { error: () => i18n.t("tagManagement.createModal.validation.nameRequired") }),
  description: z.string().optional(),
  allowed_llms: z.array(z.string()).optional(),
  max_budget: z.string().optional(),
  budget_duration: z.string().optional(),
  tpm_limit: optionalLimit,
  rpm_limit: optionalLimit,
};

const createTagSchema = z.object(createTagShape);

const budgetFieldNames = ["max_budget", "budget_duration", "tpm_limit", "rpm_limit"] as const;

const withoutBudgetFields = (values: CreateTagFormValues): CreateTagFormValues => ({
  ...values,
  max_budget: undefined,
  budget_duration: undefined,
  tpm_limit: undefined,
  rpm_limit: undefined,
});

const onlyBudgetFieldErrors = (errors: FieldErrors<CreateTagFormValues>): boolean =>
  Object.keys(errors).every((name) => (budgetFieldNames as readonly string[]).includes(name));

export type CreateTagFormValues = z.output<typeof createTagSchema>;

interface CreateTagModalProps {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (values: CreateTagFormValues) => void;
  availableModels: ModelInfo[];
}

const CreateTagModal: React.FC<CreateTagModalProps> = ({ visible, onCancel, onSubmit, availableModels }) => {
  const { t } = useTranslation();
  const [budgetSectionOpen, setBudgetSectionOpen] = React.useState(false);
  const form = useZodForm(createTagSchema, { defaultValues: { tag_name: "" } });

  const modelOptions = availableModels.map((model) => ({
    label: model.model_name,
    value: model.model_info.id,
    description: model.model_info.id,
  }));

  const handleFinish = (values: CreateTagFormValues) => {
    onSubmit(budgetSectionOpen ? values : withoutBudgetFields(values));
    form.reset();
    setBudgetSectionOpen(false);
  };

  const handleInvalid = (errors: FieldErrors<CreateTagFormValues>) => {
    if (budgetSectionOpen || !onlyBudgetFieldErrors(errors)) return;
    form.clearErrors([...budgetFieldNames]);
    handleFinish(form.getValues());
  };

  const handleCancel = () => {
    form.reset();
    onCancel();
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{t("tagManagement.createModal.title")}</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleFinish, handleInvalid)} noValidate>
          <TooltipProvider>
            <FieldGroup>
              <FormField control={form.control} name="tag_name" label={t("tagManagement.createModal.fields.name")}>
                {({ ref, ...field }) => <Input {...field} ref={ref} />}
              </FormField>

              <FormField
                control={form.control}
                name="description"
                label={t("tagManagement.createModal.fields.description")}
              >
                {({ ref, value, ...field }) => <Textarea {...field} ref={ref} value={value ?? ""} rows={4} />}
              </FormField>

              <FormField
                control={form.control}
                name="allowed_llms"
                label={labelWithHint(
                  t("tagManagement.createModal.fields.models"),
                  t("tagManagement.createModal.fields.modelsHint"),
                )}
              >
                {({ value, onChange }) => (
                  <MultiSelect
                    options={modelOptions}
                    value={value}
                    onValueChange={onChange}
                    placeholder={t("tagManagement.createModal.fields.modelsPlaceholder")}
                  />
                )}
              </FormField>
            </FieldGroup>

            <Collapsible
              open={budgetSectionOpen}
              onOpenChange={setBudgetSectionOpen}
              className="mt-4 mb-4 rounded-md border border-border"
            >
              <CollapsibleTrigger className="group flex w-full items-center justify-between px-4 py-3 text-base font-medium text-foreground">
                {t("tagManagement.createModal.budgetSection")}
                <ChevronRight className="size-4 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
              </CollapsibleTrigger>
              <CollapsibleContent className="px-4 pb-4">
                <FieldGroup className="mt-4">
                  <FormField
                    control={form.control}
                    name="max_budget"
                    label={labelWithHint(
                      t("tagManagement.createModal.fields.maxBudget"),
                      t("tagManagement.createModal.fields.maxBudgetHint"),
                    )}
                  >
                    {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={0.01} />}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="budget_duration"
                    label={labelWithHint(
                      t("tagManagement.createModal.fields.resetBudget"),
                      t("tagManagement.createModal.fields.resetBudgetHint"),
                    )}
                  >
                    {({ id, value, onChange }) => (
                      <BudgetDurationDropdown
                        id={id}
                        value={value ?? null}
                        onChange={(next) => onChange(next ?? undefined)}
                      />
                    )}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="tpm_limit"
                    label={labelWithHint(
                      t("tagManagement.createModal.fields.tpmLimit"),
                      t("tagManagement.createModal.fields.tpmLimitHint"),
                    )}
                  >
                    {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={1} min={0} />}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="rpm_limit"
                    label={labelWithHint(
                      t("tagManagement.createModal.fields.rpmLimit"),
                      t("tagManagement.createModal.fields.rpmLimitHint"),
                    )}
                  >
                    {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={1} min={0} />}
                  </FormField>
                </FieldGroup>
              </CollapsibleContent>
            </Collapsible>

            <div className="mt-2.5 text-right">
              <Button type="submit">{t("tagManagement.createModal.submit")}</Button>
            </div>
          </TooltipProvider>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateTagModal;
