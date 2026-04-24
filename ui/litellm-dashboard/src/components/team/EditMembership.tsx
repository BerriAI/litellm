import React, { useEffect, useMemo, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";
import NumericalInput from "../shared/numerical_input";

interface BaseMember {
  user_email?: string;
  user_id?: string;
  role: string;
}

interface AdditionalField {
  name: string;
  label: string | React.ReactNode;
  type: "input" | "select" | "numerical" | "multi-select";
  options?: Array<{ label: string; value: string }>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  rules?: any[];
  step?: number;
  min?: number;
  placeholder?: string;
}

interface ModalConfig {
  title: string;
  roleOptions: Array<{
    label: string;
    value: string;
  }>;
  defaultRole?: string;
  showEmail?: boolean;
  showUserId?: boolean;
  additionalFields?: AdditionalField[];
}

interface MemberModalProps<T extends BaseMember> {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (data: T) => void;
  initialData?: T | null;
  mode: "add" | "edit";
  config: ModalConfig;
}

type FormValues = {
  user_email?: string;
  user_id?: string;
  role: string;
  max_budget_in_team?: number | string | null;
  tpm_limit?: number | string | null;
  rpm_limit?: number | string | null;
  allowed_models?: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
};

function MultiSelectField({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: Array<{ label: string; value: string }>;
  placeholder?: string;
}) {
  const selected = value ?? [];
  const remaining = options.filter((o) => !selected.includes(o.value));

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder || "Select options"} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">No more options</div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge key={v} variant="secondary" className="flex items-center gap-1">
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

const MemberModal = <T extends BaseMember>({
  visible,
  onCancel,
  onSubmit,
  initialData,
  mode,
  config,
}: MemberModalProps<T>) => {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const defaultValues = useMemo<FormValues>(
    () => ({
      user_email: "",
      user_id: "",
      role: config.defaultRole || config.roleOptions[0]?.value || "",
      max_budget_in_team: null,
      tpm_limit: null,
      rpm_limit: null,
      allowed_models: [],
    }),
    [config.defaultRole, config.roleOptions],
  );

  const {
    register,
    control,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<FormValues>({
    defaultValues,
  });

  useEffect(() => {
    if (!visible) return;
    if (mode === "edit" && initialData) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const d = initialData as any;
      reset({
        ...initialData,
        role: initialData.role || config.defaultRole || "",
        max_budget_in_team: d.max_budget_in_team ?? null,
        tpm_limit: d.tpm_limit ?? null,
        rpm_limit: d.rpm_limit ?? null,
        allowed_models: d.allowed_models ?? [],
      });
    } else {
      reset(defaultValues);
    }
  }, [visible, initialData, mode, reset, config.defaultRole, defaultValues]);

  const submit = handleSubmit(async (values) => {
    try {
      setIsSubmitting(true);
      const formData = Object.entries(values).reduce((acc, [key, value]) => {
        if (typeof value === "string") {
          const trimmedValue = value.trim();
          if (trimmedValue === "" && (key === "max_budget_in_team" || key === "tpm_limit" || key === "rpm_limit")) {
            return { ...acc, [key]: null };
          }
          return { ...acc, [key]: trimmedValue };
        }
        return { ...acc, [key]: value };
      }, {}) as T;

      await Promise.resolve(onSubmit(formData));
      reset(defaultValues);
    } catch (error) {
      console.error("Form submission error:", error);
    } finally {
      setIsSubmitting(false);
    }
  });

  const getRoleLabel = (value: string) => {
    return config.roleOptions.find((option) => option.value === value)?.label || value;
  };

  const orderedRoleOptions = useMemo(() => {
    if (mode === "edit" && initialData) {
      return [
        ...config.roleOptions.filter((o) => o.value === initialData.role),
        ...config.roleOptions.filter((o) => o.value !== initialData.role),
      ];
    }
    return config.roleOptions;
  }, [mode, initialData, config.roleOptions]);

  const renderField = (field: AdditionalField) => {
    switch (field.type) {
      case "input":
        return <Input id={`field-${field.name}`} placeholder={field.placeholder} {...register(field.name)} />;
      case "numerical":
        return (
          <Controller
            control={control}
            name={field.name}
            render={({ field: rhfField }) => (
              <NumericalInput
                id={`field-${field.name}`}
                step={field.step || 1}
                min={field.min || 0}
                style={{ width: "100%" }}
                placeholder={field.placeholder || "Enter a numerical value"}
                value={rhfField.value as number | string | null | undefined}
                onChange={(v: number | null) => rhfField.onChange(v)}
              />
            )}
          />
        );
      case "select":
        return (
          <Controller
            control={control}
            name={field.name}
            render={({ field: rhfField }) => (
              <Select value={(rhfField.value as string) || ""} onValueChange={(v) => rhfField.onChange(v)}>
                <SelectTrigger id={`field-${field.name}`}>
                  <SelectValue placeholder={field.placeholder} />
                </SelectTrigger>
                <SelectContent>
                  {field.options?.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          />
        );
      case "multi-select":
        return (
          <Controller
            control={control}
            name={field.name}
            render={({ field: rhfField }) => (
              <MultiSelectField
                value={(rhfField.value as string[]) || []}
                onChange={(v) => rhfField.onChange(v)}
                options={field.options || []}
                placeholder={field.placeholder}
              />
            )}
          />
        );
      default:
        return null;
    }
  };

  return (
    <Dialog open={visible} onOpenChange={(o) => (!o ? onCancel() : undefined)}>
      <DialogContent className="max-w-[1000px]">
        <DialogHeader>
          <DialogTitle>{config.title || (mode === "add" ? "Add Member" : "Edit Member")}</DialogTitle>
        </DialogHeader>
        <form onSubmit={submit}>
          {config.showEmail && (
            <div className="grid grid-cols-3 gap-4 items-start mb-4">
              <Label htmlFor="user_email" className="pt-2">
                Email
              </Label>
              <div className="col-span-2 space-y-1">
                <Input
                  id="user_email"
                  placeholder="user@example.com"
                  {...register("user_email", {
                    validate: (value) => {
                      if (!value) return true;
                      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                      return emailRegex.test(value) || "Please enter a valid email!";
                    },
                  })}
                />
                {errors.user_email && <p className="text-sm text-destructive">{errors.user_email.message as string}</p>}
              </div>
            </div>
          )}

          {config.showEmail && config.showUserId && (
            <div className="text-center mb-4">
              <span>OR</span>
            </div>
          )}

          {config.showUserId && (
            <div className="grid grid-cols-3 gap-4 items-start mb-4">
              <Label htmlFor="user_id" className="pt-2">
                User ID
              </Label>
              <div className="col-span-2">
                <Input id="user_id" placeholder="user_123" {...register("user_id")} />
              </div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-4 items-start mb-4">
            <Label htmlFor="role" className="pt-2">
              <div className="flex items-center gap-2">
                <span>Role</span>
                {mode === "edit" && initialData && (
                  <span className="text-muted-foreground text-sm">(Current: {getRoleLabel(initialData.role)})</span>
                )}
              </div>
            </Label>
            <div className="col-span-2 space-y-1">
              <Controller
                control={control}
                name="role"
                rules={{ required: "Please select a role!" }}
                render={({ field }) => (
                  <Select value={field.value || ""} onValueChange={(v) => field.onChange(v)}>
                    <SelectTrigger id="role">
                      <SelectValue placeholder="Select a role" />
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
              />
              {errors.role && <p className="text-sm text-destructive">{errors.role.message as string}</p>}
            </div>
          </div>

          {config.additionalFields?.map((field) => (
            <div key={field.name} className="grid grid-cols-3 gap-4 items-start mb-4">
              <Label htmlFor={`field-${field.name}`} className="pt-2">
                {field.label}
              </Label>
              <div className="col-span-2">{renderField(field)}</div>
            </div>
          ))}

          <div className="text-right mt-6">
            <Button type="button" variant="outline" onClick={onCancel} className="mr-2" disabled={isSubmitting}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
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
