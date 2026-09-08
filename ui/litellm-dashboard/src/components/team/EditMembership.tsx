import React, { useEffect, useMemo, useState } from "react";
import { z } from "zod/v4";
import NumericalInput from "../shared/numerical_input";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  buildMemberFormData,
  buildMemberFormValues,
  emptyMemberFormValues,
  type MemberAdditionalField,
  type MemberFieldsConfig,
  type MemberFormValues,
} from "./memberFormValues";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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

const isEmailish = (value: string): boolean => value === "" || z.email().safeParse(value).success;

const memberFieldSchema = z.union([z.string(), z.number(), z.null(), z.array(z.string())]).optional();

const buildMemberSchema = (config: ModalConfig): z.ZodType<MemberFormValues, MemberFormValues> => {
  const shape = {
    user_email: z.string().refine(isEmailish, "Please enter a valid email!").nullish(),
    user_id: z.string().nullish(),
    role: z.string({ error: ROLE_REQUIRED_MESSAGE }).min(1, ROLE_REQUIRED_MESSAGE),
    ...Object.fromEntries((config.additionalFields ?? []).map((field) => [field.name, memberFieldSchema])),
  };

  return z.object(shape);
};

const MemberModal = <T extends BaseMember>({
  visible,
  onCancel,
  onSubmit,
  initialData,
  mode,
  config,
}: MemberModalProps<T>) => {
  const schema = useMemo(() => buildMemberSchema(config), [config]);
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
                placeholder={field.placeholder || "Enter a numerical value"}
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
                placeholder={field.placeholder || "Select options"}
              />
            );
          case "budget-duration":
            return (
              <BudgetDurationDropdown
                id={id}
                value={typeof value === "string" ? value : null}
                onChange={(next) => onChange(next)}
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
          <DialogTitle>{config.title || (mode === "add" ? "Add Member" : "Edit Member")}</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          <FieldGroup>
            {config.showEmail && (
              <FormField control={form.control} name="user_email" label="Email">
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
              <div className="text-center text-sm text-muted-foreground">OR</div>
            )}

            {config.showUserId && (
              <FormField control={form.control} name="user_id" label="User ID">
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
                  <span>Role</span>
                  {mode === "edit" && initialData && (
                    <span className="text-sm text-muted-foreground">(Current: {getRoleLabel(initialData.role)})</span>
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
              Cancel
            </Button>
            <Button type="submit" variant="outline" disabled={isSubmitting}>
              {isSubmitting && <UiLoadingSpinner className="size-4" />}
              {mode === "add"
                ? isSubmitting
                  ? "Adding..."
                  : "Add Member"
                : isSubmitting
                  ? "Saving..."
                  : "Save Changes"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default MemberModal;
