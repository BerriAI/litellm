import { useLanguage } from "@/contexts/LanguageContext";
import React, { useEffect, useEffectEvent, useMemo, useState } from "react";
import type { FieldError } from "react-hook-form";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import { z } from "zod";
import NumericalInput from "../shared/numerical_input";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { UtcDateTimeInput } from "@/components/shared/form/UtcDateTimeInput";
import { ModelMaxBudgetEditor } from "../key_team_helpers/ModelMaxBudgetEditor";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  buildMemberFormData,
  buildMemberFormValues,
  emptyMemberFormValues,
  isModelMaxBudget,
  TEMP_BUDGET_PAIR_MESSAGE,
  tempBudgetPairError,
  type MemberAdditionalField,
  type MemberFieldsConfig,
  type MemberFormValues,
} from "./memberFormValues";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

dayjs.extend(utc);

interface BaseMember {
  user_email?: string;
  user_id?: string;
  role: string;
}

interface ModalConfig extends MemberFieldsConfig {
  title: string;
}

interface MemberModalProps<T extends BaseMember> {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (data: T) => void;
  initialData?: T | null;
  mode: "add" | "edit";
  config: ModalConfig;
}

const ROLE_REQUIRED_MESSAGE = "Please select a role!";
const EMAIL_INVALID_MESSAGE = "Please enter a valid email!";
const LOCAL_VALIDATION_MESSAGES = new Set([ROLE_REQUIRED_MESSAGE, EMAIL_INVALID_MESSAGE, TEMP_BUDGET_PAIR_MESSAGE]);

const isEmailish = (value: string): boolean =>
  value === "" || z.email({ pattern: z.regexes.html5Email }).safeParse(value).success;

const modelMaxBudgetSchema = z.record(z.string(), z.looseObject({ budget_limit: z.number(), time_period: z.string() }));

const memberFieldSchema = z
  .union([z.string(), z.number(), z.null(), z.array(z.string()), modelMaxBudgetSchema])
  .optional();

const buildMemberSchema = (config: ModalConfig): z.ZodType<MemberFormValues, MemberFormValues> => {
  const shape = {
    user_email: z.string().refine(isEmailish, EMAIL_INVALID_MESSAGE).nullish(),
    user_id: z.string().nullish(),
    role: z.string({ error: ROLE_REQUIRED_MESSAGE }).min(1, ROLE_REQUIRED_MESSAGE),
    ...Object.fromEntries((config.additionalFields ?? []).map((field) => [field.name, memberFieldSchema])),
  };

  return z.object(shape).superRefine((values, ctx) => {
    const path = tempBudgetPairError(values);
    if (path !== null) ctx.addIssue({ code: "custom", path: [path], message: TEMP_BUDGET_PAIR_MESSAGE });
  });
};

const MemberModal = <T extends BaseMember>({
  visible,
  onCancel,
  onSubmit,
  initialData,
  mode,
  config,
}: MemberModalProps<T>) => {
  const { t } = useLanguage();
  const schema = useMemo(() => buildMemberSchema(config), [config]);
  const form = useZodForm(schema, { defaultValues: emptyMemberFormValues(config) });
  const [isSubmitting, setIsSubmitting] = useState(false);

  // 只在显示时翻译本地校验原文，已有错误随语言刷新，业务数据和未知错误不变。
  const renderValidationError = (error: FieldError) =>
    error.message && LOCAL_VALIDATION_MESSAGES.has(error.message) ? t(error.message) : error.message;

  // 翻译后的配置标签会改变引用，仅打开弹窗或切换成员时初始化字段。
  const resetMemberForm = useEffectEvent(() => {
    form.reset(buildMemberFormValues(mode, initialData as MemberFormValues | null | undefined, config));
  });

  useEffect(() => {
    if (visible) {
      resetMemberForm();
    }
  }, [visible, initialData, mode, form]);

  const handleSubmit = async (values: MemberFormValues) => {
    try {
      setIsSubmitting(true);
      await Promise.resolve(onSubmit(buildMemberFormData(values) as unknown as T));
      form.reset(emptyMemberFormValues(config));
    } catch (error) {
      console.error("Form submission error:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getRoleLabel = (value: string) => config.roleOptions.find((option) => option.value === value)?.label || value;

  const orderedRoleOptions =
    mode === "edit" && initialData
      ? [
          ...config.roleOptions.filter((option) => option.value === initialData.role),
          ...config.roleOptions.filter((option) => option.value !== initialData.role),
        ]
      : config.roleOptions;

  const renderField = (field: MemberAdditionalField, name: string) => (
    <FormField key={name} control={form.control} name={name} label={field.label} renderError={renderValidationError}>
      {({ ref, id, value, onChange, ...rest }) => {
        switch (field.type) {
          case "input":
            return (
              <Input
                {...rest}
                id={id}
                ref={ref}
                placeholder={field.placeholder}
                value={typeof value === "string" ? value : ""}
                onChange={(event) => onChange(event.target.value)}
              />
            );
          case "numerical":
            return (
              <NumericalInput
                {...rest}
                id={id}
                step={field.step || 1}
                min={field.min || 0}
                style={{ width: "100%" }}
                placeholder={field.placeholder || t("Enter a numerical value")}
                value={value ?? ""}
                onChange={(event: React.ChangeEvent<HTMLInputElement>) => onChange(event.target.value)}
              />
            );
          case "select":
            return (
              <Select
                items={Object.fromEntries((field.options ?? []).map((option) => [option.value, option.label]))}
                value={typeof value === "string" && value !== "" ? value : null}
                onValueChange={(selected: string | null) => onChange(selected ?? undefined)}
              >
                <SelectTrigger id={id} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {field.options?.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            );
          case "multi-select":
            return (
              <MultiSelect
                options={field.options ?? []}
                value={Array.isArray(value) ? value : []}
                onValueChange={onChange}
                placeholder={field.placeholder || t("Select options")}
              />
            );
          case "budget-duration":
            return (
              <BudgetDurationDropdown
                id={id}
                value={typeof value === "string" ? value : null}
                onChange={(next) => onChange(mode === "add" ? next ?? undefined : next)}
              />
            );
          case "model-max-budget":
            return (
              <ModelMaxBudgetEditor
                key={`${visible}-${initialData?.user_id ?? initialData?.user_email ?? ""}`}
                value={isModelMaxBudget(value) ? value : {}}
                onChange={onChange}
                availableModels={field.availableModels ?? []}
                premiumUser={field.premiumUser ?? false}
              />
            );
          case "utc-datetime":
            return (
              <UtcDateTimeInput
                {...rest}
                id={id}
                ref={ref}
                value={typeof value === "string" && value !== "" ? dayjs.utc(value) : null}
                onChange={(next) => onChange(next === null ? null : next.toISOString())}
              />
            );
          default:
            return null;
        }
      }}
    </FormField>
  );

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
        <DialogHeader>
          <DialogTitle>{config.title || (mode === "add" ? t("Add Member") : t("Edit Member"))}</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          <FieldGroup>
            {config.showEmail && (
              <FormField
                control={form.control}
                name="user_email"
                label={t("Email")}
                renderError={renderValidationError}
              >
                {({ ref, value, onChange, ...rest }) => (
                  <Input
                    {...rest}
                    ref={ref}
                    placeholder="user@example.com"
                    value={typeof value === "string" ? value : ""}
                    onChange={(event) => onChange(event.target.value)}
                  />
                )}
              </FormField>
            )}

            {config.showEmail && config.showUserId && (
              <div className="text-center text-sm text-muted-foreground">{t("OR")}</div>
            )}

            {config.showUserId && (
              <FormField control={form.control} name="user_id" label={t("User ID")}>
                {({ ref, value, onChange, ...rest }) => (
                  <Input
                    {...rest}
                    ref={ref}
                    placeholder="user_123"
                    value={typeof value === "string" ? value : ""}
                    onChange={(event) => onChange(event.target.value)}
                  />
                )}
              </FormField>
            )}

            <FormField
              control={form.control}
              name="role"
              renderError={renderValidationError}
              label={
                <span className="flex items-center gap-2">
                  <span>{t("Role")}</span>
                  {mode === "edit" && initialData && (
                    <span className="text-sm text-muted-foreground">
                      {t("(Current: {role})", { role: getRoleLabel(initialData.role) })}
                    </span>
                  )}
                </span>
              }
            >
              {({ id, value, onChange }) => (
                <Select
                  items={Object.fromEntries(orderedRoleOptions.map((option) => [option.value, option.label]))}
                  value={typeof value === "string" && value !== "" ? value : null}
                  onValueChange={(selected: string | null) => onChange(selected ?? undefined)}
                >
                  <SelectTrigger id={id} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {orderedRoleOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>

            {config.additionalFields?.map((field) => renderField(field, field.name))}
          </FieldGroup>

          <div className="mt-6 text-right">
            <Button type="button" variant="outline" onClick={onCancel} disabled={isSubmitting} className="mr-2">
              {t("Cancel")}
            </Button>
            <Button type="submit" variant="outline" disabled={isSubmitting}>
              {isSubmitting && <UiLoadingSpinner className="size-4" />}
              {mode === "add"
                ? isSubmitting
                  ? t("Adding...")
                  : t("Add Member")
                : isSubmitting
                  ? t("Saving...")
                  : t("Save Changes")}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default MemberModal;
