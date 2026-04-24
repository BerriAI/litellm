import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Info, X } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { all_admin_roles } from "../utils/roles";
import BudgetDurationDropdown from "./common_components/budget_duration_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import NumericalInput from "./shared/numerical_input";

interface UserEditViewProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  userData: any;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onSubmit: (values: any) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  teams: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  userModels: string[];
  possibleUIRoles: Record<string, Record<string, string>> | null;
  isBulkEdit?: boolean;
}

interface UserEditFormValues {
  user_id?: string;
  user_email?: string;
  user_alias?: string;
  user_role?: string;
  models: string[];
  max_budget: number | string | null;
  budget_duration?: string | null;
  metadata?: string;
}

/**
 * Multi-select with shadcn Select + badge chip list. Mirrors the pattern
 * used in `AccessGroupBaseForm.tsx`.
 */
function ModelMultiSelect({
  value,
  onChange,
  options,
  disabled,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  disabled?: boolean;
  placeholder: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
        disabled={disabled}
      >
        <SelectTrigger aria-label={placeholder}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No more options
            </div>
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
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                {!disabled && (
                  <button
                    type="button"
                    onClick={() => onChange(selected.filter((s) => s !== v))}
                    className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                    aria-label={`Remove ${opt?.label ?? v}`}
                  >
                    <X size={12} />
                  </button>
                )}
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function UserEditView({
  userData,
  onCancel,
  onSubmit,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  teams,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  accessToken,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userID,
  userRole,
  userModels,
  possibleUIRoles,
  isBulkEdit = false,
}: UserEditViewProps) {
  const [unlimitedBudget, setUnlimitedBudget] = useState(false);
  const {
    control,
    register,
    handleSubmit,
    setValue,
    reset,
    formState: { errors },
  } = useForm<UserEditFormValues>({
    defaultValues: {
      user_id: userData.user_id,
      user_email: userData.user_info?.user_email,
      user_alias: userData.user_info?.user_alias,
      user_role: userData.user_info?.user_role,
      models: userData.user_info?.models || [],
      max_budget:
        userData.user_info?.max_budget === null ||
        userData.user_info?.max_budget === undefined
          ? ""
          : userData.user_info?.max_budget,
      budget_duration: userData.user_info?.budget_duration,
      metadata: userData.user_info?.metadata
        ? JSON.stringify(userData.user_info.metadata, null, 2)
        : undefined,
    },
  });

  useEffect(() => {
    const maxBudget = userData.user_info?.max_budget;
    const isUnlimited = maxBudget === null || maxBudget === undefined;
    setUnlimitedBudget(isUnlimited);

    reset({
      user_id: userData.user_id,
      user_email: userData.user_info?.user_email,
      user_alias: userData.user_info?.user_alias,
      user_role: userData.user_info?.user_role,
      models: userData.user_info?.models || [],
      max_budget: isUnlimited ? "" : maxBudget,
      budget_duration: userData.user_info?.budget_duration,
      metadata: userData.user_info?.metadata
        ? JSON.stringify(userData.user_info.metadata, null, 2)
        : undefined,
    });
  }, [userData, reset]);

  const handleUnlimitedBudgetChange = (checked: boolean) => {
    setUnlimitedBudget(checked);
    if (checked) {
      setValue("max_budget", "");
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const submitHandler = (values: UserEditFormValues) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const out: Record<string, any> = { ...values };
    if (out.metadata && typeof out.metadata === "string") {
      try {
        out.metadata = JSON.parse(out.metadata);
      } catch (error) {
        console.error("Error parsing metadata JSON:", error);
        return;
      }
    }

    if (
      unlimitedBudget ||
      out.max_budget === "" ||
      out.max_budget === undefined
    ) {
      out.max_budget = null;
    }

    onSubmit(out);
  };

  const modelOptions = useMemo(() => {
    const base = [
      { label: "All Proxy Models", value: "all-proxy-models" },
      { label: "No Default Models", value: "no-default-models" },
    ];
    const extras = userModels.map((model) => ({
      label: getModelDisplayName(model),
      value: model,
    }));
    return [...base, ...extras];
  }, [userModels]);

  const modelsDisabled = !all_admin_roles.includes(userRole || "");
  const possibleRoleEntries = possibleUIRoles
    ? Object.entries(possibleUIRoles)
    : [];

  return (
    <form onSubmit={handleSubmit(submitHandler)} className="space-y-4">
      {!isBulkEdit && (
        <div className="space-y-2">
          <Label htmlFor="user_id">User ID</Label>
          <Input id="user_id" disabled {...register("user_id")} />
        </div>
      )}

      {!isBulkEdit && (
        <div className="space-y-2">
          <Label htmlFor="user_email">Email</Label>
          <Input id="user_email" {...register("user_email")} />
        </div>
      )}

      <div className="space-y-2">
        <Label htmlFor="user_alias">User Alias</Label>
        <Input id="user_alias" {...register("user_alias")} />
      </div>

      <div className="space-y-2">
        <Label htmlFor="user_role" className="flex items-center">
          Global Proxy Role
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                This is the role that the user will globally on the proxy. This
                role is independent of any team/org specific roles.
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </Label>
        <Controller
          control={control}
          name="user_role"
          render={({ field }) => (
            <Select
              value={field.value ?? ""}
              onValueChange={(v) => field.onChange(v)}
            >
              <SelectTrigger id="user_role">
                <SelectValue placeholder="Select a role" />
              </SelectTrigger>
              <SelectContent>
                {possibleRoleEntries.map(
                  ([role, { ui_label, description }]) => (
                    <SelectItem key={role} value={role} title={ui_label}>
                      <div className="flex">
                        {ui_label}
                        <p className="ml-2 text-muted-foreground text-xs">
                          {description}
                        </p>
                      </div>
                    </SelectItem>
                  ),
                )}
              </SelectContent>
            </Select>
          )}
        />
      </div>

      <div className="space-y-2">
        <Label className="flex items-center">
          Personal Models
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Select which models this user can access outside of team-scope.
                Choose &apos;All Proxy Models&apos; to grant access to all
                models available on the proxy.
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </Label>
        <Controller
          control={control}
          name="models"
          render={({ field }) => (
            <ModelMultiSelect
              value={field.value ?? []}
              onChange={field.onChange}
              options={modelOptions}
              disabled={modelsDisabled}
              placeholder="Select models"
            />
          )}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <Label htmlFor="max_budget">Max Budget (USD)</Label>
          <label className="flex items-center gap-2 cursor-pointer">
            <Checkbox
              checked={unlimitedBudget}
              onCheckedChange={(c) => handleUnlimitedBudgetChange(c === true)}
            />
            <span className="text-sm">Unlimited Budget</span>
          </label>
        </div>
        <Controller
          control={control}
          name="max_budget"
          rules={{
            validate: (value) => {
              if (
                !unlimitedBudget &&
                (value === "" || value === null || value === undefined)
              ) {
                return "Please enter a budget or select Unlimited Budget";
              }
              return true;
            },
          }}
          render={({ field }) => (
            <NumericalInput
              id="max_budget"
              step={0.01}
              precision={2}
              style={{ width: "100%" }}
              disabled={unlimitedBudget}
              value={field.value as number | string | null | undefined}
              onChange={(v: number | null) => field.onChange(v === null ? "" : v)}
            />
          )}
        />
        {errors.max_budget && (
          <p className="text-sm text-destructive">
            {errors.max_budget.message as string}
          </p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="budget_duration">Reset Budget</Label>
        <Controller
          control={control}
          name="budget_duration"
          render={({ field }) => (
            <BudgetDurationDropdown
              value={field.value ?? null}
              onChange={(v) => field.onChange(v)}
            />
          )}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="metadata">Metadata</Label>
        <Textarea
          id="metadata"
          rows={4}
          placeholder="Enter metadata as JSON"
          {...register("metadata")}
        />
      </div>

      <div className="flex justify-end space-x-2">
        <Button variant="secondary" type="button" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">Save Changes</Button>
      </div>
    </form>
  );
}
