"use client";

import type { ParseKeys } from "i18next";
import { Plus, X } from "lucide-react";
import React from "react";
import { useController } from "react-hook-form";
import { Trans, useTranslation } from "react-i18next";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  asText,
  GuardrailField,
  labelWithHint,
  requiredRule,
  type GuardrailCriterion,
  type GuardrailFieldControlProps,
  type GuardrailFormControl,
} from "../GuardrailFormField";

interface LLMJudgeFieldsProps {
  availableModels: string[];
  control: GuardrailFormControl;
}

const DEFAULT_CRITERIA: GuardrailCriterion[] = [{ name: "", weight: 100, description: "" }];

const ON_FAILURE_ITEMS: { labelKey: ParseKeys; value: string }[] = [
  { labelKey: "guardrails.llmJudge.onFailure.options.block", value: "block" },
  { labelKey: "guardrails.llmJudge.onFailure.options.log", value: "log" },
];

const clampToRange = (value: unknown, min: number, max: number): number | null => {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  return Math.min(max, Math.max(min, value));
};

interface BoundedNumberInputProps {
  control: GuardrailFieldControlProps;
  min: number;
  max: number;
  suffix: string;
  placeholder?: string;
}

const BoundedNumberInput: React.FC<BoundedNumberInputProps> = ({ control, min, max, suffix, placeholder }) => {
  const { id, name, value, onChange, onBlur, ...aria } = control;

  return (
    <InputGroup>
      <InputGroupInput
        id={id}
        name={name}
        type="number"
        min={min}
        max={max}
        placeholder={placeholder}
        value={asText(value)}
        onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))}
        onBlur={() => {
          onChange(clampToRange(value, min, max));
          onBlur();
        }}
        {...aria}
      />
      <InputGroupAddon align="inline-end">{suffix}</InputGroupAddon>
    </InputGroup>
  );
};

const LLMJudgeFields: React.FC<LLMJudgeFieldsProps> = ({ availableModels, control }) => {
  const { t } = useTranslation();
  const { field } = useController({ control, name: "criteria", defaultValue: DEFAULT_CRITERIA });
  const criteria: GuardrailCriterion[] = Array.isArray(field.value) ? field.value : [];
  const setCriteria = field.onChange;

  const weightTotal = criteria.reduce((sum, entry) => sum + (Number(entry?.weight) || 0), 0);
  const weightOk = weightTotal === 100;
  const onFailureItems = ON_FAILURE_ITEMS.map(({ labelKey, value }) => ({ label: t(labelKey), value }));

  return (
    <FieldGroup>
      <div className="rounded-md border border-success/20 bg-success/10 px-3.5 py-2.5 text-[13px] text-success">
        <Trans i18nKey="guardrails.llmJudge.intro" components={{ strong: <strong /> }} />
      </div>

      <GuardrailField
        control={control}
        name="judge_model"
        label={labelWithHint(t("guardrails.llmJudge.judgeModel.label"), t("guardrails.llmJudge.judgeModel.hint"))}
        rules={requiredRule(t("guardrails.llmJudge.judgeModel.required"))}
      >
        {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
          <Combobox items={availableModels} value={asText(value) || null} onValueChange={onChange}>
            <ComboboxInput
              id={id}
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
              placeholder={t("guardrails.llmJudge.judgeModel.placeholder")}
              className="w-full"
            />
            <ComboboxContent>
              <ComboboxEmpty>{t("guardrails.llmJudge.judgeModel.empty")}</ComboboxEmpty>
              <ComboboxList>
                {(model: string) => (
                  <ComboboxItem key={model} value={model} title={model}>
                    {model}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        )}
      </GuardrailField>

      <GuardrailField
        control={control}
        name="overall_threshold"
        label={labelWithHint(t("guardrails.llmJudge.threshold.label"), t("guardrails.llmJudge.threshold.hint"))}
        defaultValue={80}
      >
        {(fieldControl) => <BoundedNumberInput control={fieldControl} min={0} max={100} suffix="/ 100" />}
      </GuardrailField>

      <GuardrailField
        control={control}
        name="on_failure"
        label={labelWithHint(t("guardrails.llmJudge.onFailure.label"), t("guardrails.llmJudge.onFailure.hint"))}
        defaultValue="block"
      >
        {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
          <Select items={onFailureItems} value={asText(value) || null} onValueChange={onChange}>
            <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} className="w-full">
              <SelectValue placeholder={t("guardrails.llmJudge.onFailure.placeholder")} />
            </SelectTrigger>
            <SelectContent>
              {onFailureItems.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </GuardrailField>

      <Field>
        <FieldLabel>
          {labelWithHint(t("guardrails.llmJudge.criteria.label"), t("guardrails.llmJudge.criteria.hint"))}
        </FieldLabel>

        {criteria.map((_, index) => (
          <div key={index} className="mb-2 rounded-md border border-border p-3">
            <div className="flex items-end gap-2">
              <GuardrailField
                control={control}
                name={`criteria.${index}.name`}
                rules={requiredRule(t("guardrails.llmJudge.criteria.nameRequired"))}
                className="flex-2"
              >
                {({ ref, value, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    value={asText(value)}
                    placeholder={t("guardrails.llmJudge.criteria.namePlaceholder")}
                  />
                )}
              </GuardrailField>
              <GuardrailField
                control={control}
                name={`criteria.${index}.weight`}
                label={labelWithHint(
                  <span className="text-xs text-muted-foreground">
                    {t("guardrails.llmJudge.criteria.weightLabel")}
                  </span>,
                  t("guardrails.llmJudge.criteria.weightHint"),
                )}
                rules={requiredRule(t("guardrails.llmJudge.criteria.weightRequired"))}
                className="flex-1"
              >
                {(fieldControl) => (
                  <BoundedNumberInput
                    control={fieldControl}
                    min={0}
                    max={100}
                    suffix="%"
                    placeholder={t("guardrails.llmJudge.criteria.weightPlaceholder")}
                  />
                )}
              </GuardrailField>
              <Button
                variant="ghost"
                size="sm"
                aria-label={t("guardrails.llmJudge.criteria.remove")}
                className="mb-1 text-destructive hover:text-destructive/80"
                onClick={() => setCriteria(criteria.filter((_, position) => position !== index))}
              >
                <X className="size-4" />
              </Button>
            </div>
            <GuardrailField
              control={control}
              name={`criteria.${index}.description`}
              rules={requiredRule(t("guardrails.llmJudge.criteria.descriptionRequired"))}
              className="mt-2"
            >
              {({ ref, value, ...field }) => (
                <Input
                  {...field}
                  ref={ref}
                  value={asText(value)}
                  placeholder={t("guardrails.llmJudge.criteria.descriptionPlaceholder")}
                />
              )}
            </GuardrailField>
          </div>
        ))}

        <Button
          variant="outline"
          className="mt-1 w-full border-dashed"
          onClick={() => setCriteria([...criteria, { name: "", weight: 0, description: "" }])}
        >
          <Plus className="size-4" />
          {t("guardrails.llmJudge.criteria.add")}
        </Button>

        {criteria.length > 0 && (
          <div className={`mt-1.5 text-xs ${weightOk ? "text-success" : "text-warning"}`}>
            {weightOk
              ? t("guardrails.llmJudge.criteria.weightsTotalOk", { total: weightTotal })
              : t("guardrails.llmJudge.criteria.weightsTotalOff", { total: weightTotal })}
          </div>
        )}
      </Field>
    </FieldGroup>
  );
};

export default LLMJudgeFields;
