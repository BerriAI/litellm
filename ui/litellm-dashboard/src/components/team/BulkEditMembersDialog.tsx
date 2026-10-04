"use client";

import { CircleHelp } from "lucide-react";
import React from "react";
import { Controller, useWatch, type Control } from "react-hook-form";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import type { Member } from "@/components/networking";
import { FormField } from "@/components/shared/form/FormField";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  bulkMemberLimitsSchema,
  EMPTY_BULK_MEMBER_LIMITS,
  type BulkMemberLimitField,
  type BulkMemberLimitsFormValues,
  type MemberLimitsPatch,
} from "./bulkMemberLimits";
import { MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES } from "./memberBudgetReset";

const PREVIEW_COUNT = 3;

interface BulkEditMembersDialogProps {
  members: readonly Member[] | null;
  teamModels: readonly string[];
  isSaving: boolean;
  onCancel: () => void;
  onSubmit: (limits: MemberLimitsPatch) => void;
}

const labelWithHint = (title: string, hint: string): React.ReactNode => (
  <span className="flex items-center gap-1">
    {title}
    <SimpleTooltip content={hint}>
      <CircleHelp className="size-3.5 text-muted-foreground" />
    </SimpleTooltip>
  </span>
);

type BulkMemberLimitsControl = Control<BulkMemberLimitsFormValues, unknown, MemberLimitsPatch>;

const memberName = (member: Member): string => member.user_email || member.user_id || "unknown";

function LimitRow({
  control,
  name,
  title,
  children,
}: {
  control: BulkMemberLimitsControl;
  name: BulkMemberLimitField;
  title: string;
  children: (enabled: boolean) => React.ReactNode;
}) {
  const enabled = useWatch({ control, name: `${name}.change` });
  return (
    <div className="flex items-start gap-3">
      <Controller
        control={control}
        name={`${name}.change`}
        render={({ field }) => (
          <Checkbox
            className="mt-1"
            aria-label={`Change ${title}`}
            data-testid={`bulk-change-${name}`}
            checked={field.value}
            onCheckedChange={(checked) => field.onChange(checked === true)}
          />
        )}
      />
      <div className="min-w-0 flex-1">{children(enabled)}</div>
    </div>
  );
}

function BulkEditMembersForm({
  members,
  teamModels,
  isSaving,
  onCancel,
  onSubmit,
}: Omit<BulkEditMembersDialogProps, "members"> & { members: readonly Member[] }) {
  const form = useZodForm(bulkMemberLimitsSchema, { defaultValues: EMPTY_BULK_MEMBER_LIMITS });
  const values = useWatch({ control: form.control });
  const anyChange = Object.values(values).some((field) => field?.change === true);
  const tooMany = members.length > MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES;
  const clearHint = (enabled: boolean, what: string) => (enabled ? `Leave blank to clear ${what}` : undefined);
  const modelOptions = teamModels.map((model) => ({ label: model, value: model }));
  const preview = members.slice(0, PREVIEW_COUNT).map(memberName).join(", ");
  const remaining = members.length - PREVIEW_COUNT;

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} noValidate>
      <DialogDescription>
        Tick the limits to change for {preview}
        {remaining > 0 && ` and ${remaining} more`}. Unticked limits stay as they are for each member.
      </DialogDescription>
      <FieldGroup className="mt-4">
        <LimitRow control={form.control} name="max_budget_in_team" title="Team Member Budget (USD)">
          {(enabled) => (
            <FormField
              control={form.control}
              name="max_budget_in_team.value"
              label={labelWithHint(
                "Team Member Budget (USD)",
                "Maximum amount in USD each member can spend within this team",
              )}
              description={clearHint(enabled, "the custom budget")}
            >
              {(field) => <Input {...field} type="text" inputMode="decimal" disabled={!enabled} />}
            </FormField>
          )}
        </LimitRow>
        <LimitRow control={form.control} name="budget_duration" title="Budget Reset Period">
          {(enabled) => (
            <FormField
              control={form.control}
              name="budget_duration.value"
              label={labelWithHint("Budget Reset Period", "How often each member's budget resets within the team")}
              description={clearHint(enabled, "the reset period")}
            >
              {({ id, value, onChange }) => (
                <BudgetDurationDropdown
                  id={id}
                  value={value || null}
                  onChange={(next) => onChange(next ?? "")}
                  disabled={!enabled}
                />
              )}
            </FormField>
          )}
        </LimitRow>
        <LimitRow control={form.control} name="tpm_limit" title="Team Member TPM Limit">
          {(enabled) => (
            <FormField
              control={form.control}
              name="tpm_limit.value"
              label={labelWithHint(
                "Team Member TPM Limit",
                "Maximum tokens per minute each member can use in this team",
              )}
              description={clearHint(enabled, "the TPM limit")}
            >
              {(field) => <Input {...field} type="text" inputMode="numeric" disabled={!enabled} />}
            </FormField>
          )}
        </LimitRow>
        <LimitRow control={form.control} name="rpm_limit" title="Team Member RPM Limit">
          {(enabled) => (
            <FormField
              control={form.control}
              name="rpm_limit.value"
              label={labelWithHint(
                "Team Member RPM Limit",
                "Maximum requests per minute each member can make in this team",
              )}
              description={clearHint(enabled, "the RPM limit")}
            >
              {(field) => <Input {...field} type="text" inputMode="numeric" disabled={!enabled} />}
            </FormField>
          )}
        </LimitRow>
        <LimitRow control={form.control} name="allowed_models" title="Allowed Models">
          {(enabled) => (
            <FormField
              control={form.control}
              name="allowed_models.value"
              label={labelWithHint("Allowed Models", "Models each member can access within this team")}
              description={enabled ? "Leave empty to let members use every team model" : undefined}
            >
              {({ id, value, onChange }) => (
                <MultiSelect
                  id={id}
                  options={modelOptions}
                  value={value}
                  onValueChange={onChange}
                  placeholder="All team models"
                  disabled={!enabled}
                />
              )}
            </FormField>
          )}
        </LimitRow>
      </FieldGroup>
      {tooMany && (
        <p className="mt-4 text-sm text-destructive" role="alert">
          Select at most {MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES} members at a time. {members.length} are selected.
        </p>
      )}
      <div className="mt-6 flex justify-end gap-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={isSaving}>
          Cancel
        </Button>
        <Button type="submit" disabled={!anyChange || tooMany || isSaving}>
          {isSaving ? "Saving..." : `Update ${members.length} member${members.length === 1 ? "" : "s"}`}
        </Button>
      </div>
    </form>
  );
}

export default function BulkEditMembersDialog({ members, ...props }: BulkEditMembersDialogProps) {
  return (
    <Dialog
      open={members !== null}
      onOpenChange={(open) => {
        if (!open && !props.isSaving) props.onCancel();
      }}
    >
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>Bulk Edit Members</DialogTitle>
        </DialogHeader>
        {members !== null && <BulkEditMembersForm members={members} {...props} />}
      </DialogContent>
    </Dialog>
  );
}
