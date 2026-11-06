import React, { useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
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
  TEMP_BUDGET_PAIR_MESSAGE_KEY,
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

const isEmailish = (value: string): boolean =>
  value === "" || z.email({ pattern: z.regexes.html5Email }).safeParse(value).success;

const modelMaxBudgetSchema = z.record(z.string(), z.looseObject({ budget_limit: z.number(), time_period: z.string() }));

const memberFieldSchema = z
  .union([z.string(), z.number(), z.null(), z.array(z.string()), modelMaxBudgetSchema])
  .optional();

const buildMemberSchema = (config: ModalConfig, t: TFunction): z.ZodType<MemberFormValues, MemberFormValues> => {
  const roleRequired = t("teams.editMembership.errors.roleRequired");
  const shape = {
    user_email: z.string().refine(isEmailish, t("teams.editMembership.errors.invalidEmail")).nullish(),
    user_id: z.string().nullish(),
    role: z.string({ error: roleRequired }).min(1, roleRequired),
    ...Object.fromEntries((config.additionalFields ?? []).map((field) => [field.name, memberFieldSchema])),
  };

  return z.object(shape).superRefine((values, ctx) => {
    const path = tempBudgetPairError(values);
    if (path !== null) ctx.addIssue({ code: "custom", path: [path], message: t(TEMP_BUDGET_PAIR_MESSAGE_KEY) });
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
  const { t } = useTranslation();
  const schema = useMemo(() => buildMemberSchema(config, t), [config, t]);
  const form = useZodForm(schema, { defaultValues: emptyMemberFormValues(config) });
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (visible) {
      form.reset(buildMemberFormValues(mode, initialData as MemberFormValues | null | undefined, config));
    }
  }, [visible, initialData, mode, form, config]);

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
    <FormField key={name} control={form.control} name={name} label={field.label}>
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
                placeholder={field.placeholder || t("teams.editMembership.numericalPlaceholder")}
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
                placeholder={field.placeholder || t("teams.editMembership.multiSelectPlaceholder")}
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
          <DialogTitle>
            {config.title ||
              (mode === "add" ? t("teams.editMembership.addTitle") : t("teams.editMembership.editTitle"))}
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          <FieldGroup>
            {config.showEmail && (
              <FormField control={form.control} name="user_email" label={t("teams.editMembership.fields.email")}>
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
              <div className="text-center text-sm text-muted-foreground">{t("teams.editMembership.fields.or")}</div>
            )}

            {config.showUserId && (
              <FormField control={form.control} name="user_id" label={t("teams.editMembership.fields.userId")}>
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
              label={
                <span className="flex items-center gap-2">
                  <span>{t("teams.editMembership.fields.role")}</span>
                  {mode === "edit" && initialData && (
                    <span className="text-sm text-muted-foreground">
                      {t("teams.editMembership.fields.currentRole", { role: getRoleLabel(initialData.role) })}
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
              {t("teams.editMembership.cancel")}
            </Button>
            <Button type="submit" variant="outline" disabled={isSubmitting}>
              {isSubmitting && <UiLoadingSpinner className="size-4" />}
              {mode === "add"
                ? isSubmitting
                  ? t("teams.editMembership.submit.adding")
                  : t("teams.editMembership.submit.add")
                : isSubmitting
                  ? t("teams.editMembership.submit.saving")
                  : t("teams.editMembership.submit.save")}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default MemberModal;
