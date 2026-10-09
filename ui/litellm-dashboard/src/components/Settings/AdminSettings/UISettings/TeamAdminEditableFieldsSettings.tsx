"use client";

import { CircleHelp } from "lucide-react";
import { Fragment, type ReactNode } from "react";
import { Controller } from "react-hook-form";
import { z } from "zod";

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useUpdateUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  parseSupportedTeamAdminEditableFields,
  parseTeamAdminEditableFields,
  TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION,
  teamAdminFieldLabel,
  type TeamAdminSettingsField,
} from "@/components/team/teamAdminEditAccess";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

const editableFieldsSchema = z.object({ team_admin_editable_team_fields: z.array(z.string()) });

type SaveEditableFields = ReturnType<typeof useUpdateUISettings>["mutate"];

const MAX_BUDGET_FIELD: TeamAdminSettingsField = "max_budget";

const RAISE_MAX_BUDGET_HELP_LABEL = "About raising the team's max budget";
const RAISE_MAX_BUDGET_TOOLTIP =
  "Lets team admins raise the budget too, capped by the organization's budget when the team has one. Only proxy admins can remove it.";
const RAISE_MAX_BUDGET_DOCS_URL =
  "https://docs.litellm.ai/docs/proxy/access_control#choosing-what-team-admins-can-edit";

const toggleEditableField = (
  supportedFields: readonly string[],
  draft: readonly string[],
  name: string,
  checked: boolean,
): string[] => {
  const selected = supportedFields.filter((item) => (item === name ? checked : draft.includes(item)));
  return selected.includes(MAX_BUDGET_FIELD)
    ? selected
    : selected.filter((item) => item !== TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION);
};

export default function TeamAdminEditableFieldsSettings() {
  const { accessToken } = useAuthorized();
  const { data, isLoading } = useUISettings();
  const { mutate: saveSettings, isPending } = useUpdateUISettings(accessToken);
  const supportedFields = parseSupportedTeamAdminEditableFields(data?.field_schema);
  const savedFields = parseTeamAdminEditableFields(data?.values);
  const enabledFields = supportedFields.filter((field) => savedFields.includes(field));

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardTitle>Team admin editable fields</CardTitle>
          <Badge variant={enabledFields.length > 0 ? "secondary" : "outline"}>
            {enabledFields.length > 0
              ? `${enabledFields.length} field${enabledFields.length !== 1 ? "s" : ""} enabled`
              : "Team admins cannot edit team settings"}
          </Badge>
        </div>
        <CardDescription>
          {data?.field_schema?.properties?.team_admin_editable_team_fields?.description ??
            "Team settings fields a team admin may change on the teams they administer."}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-16 w-full" />
        ) : (
          <TeamAdminEditableFieldsForm
            key={enabledFields.join(",")}
            enabledFields={enabledFields}
            supportedFields={supportedFields}
            isPending={isPending}
            saveSettings={saveSettings}
          />
        )}
      </CardContent>
    </Card>
  );
}

interface TeamAdminEditableFieldsFormProps {
  enabledFields: readonly string[];
  supportedFields: readonly string[];
  isPending: boolean;
  saveSettings: SaveEditableFields;
}

function TeamAdminEditableFieldsForm({
  enabledFields,
  supportedFields,
  isPending,
  saveSettings,
}: TeamAdminEditableFieldsFormProps) {
  const form = useZodForm(editableFieldsSchema, {
    defaultValues: { team_admin_editable_team_fields: [...enabledFields] },
  });
  const submit = form.handleSubmit((values) =>
    saveSettings(values, {
      onSuccess: () => {
        form.reset(values);
        toast.success("Team admin editable fields updated successfully");
      },
      onError: (error) => {
        toast.fromError(error);
      },
    }),
  );

  if (supportedFields.length === 0) {
    return (
      <p className="text-sm italic text-muted-foreground">
        This proxy version does not support enabling any team settings fields for team admins yet.
      </p>
    );
  }

  const raiseMaxBudgetSupported = supportedFields.includes(TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION);
  const listedFields = supportedFields.filter((name) => name !== TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION);

  return (
    <form onSubmit={(event) => void submit(event)} className="space-y-4">
      <Controller
        control={form.control}
        name="team_admin_editable_team_fields"
        render={({ field }) => {
          const toggle = (name: string) => (checked: boolean) =>
            field.onChange(toggleEditableField(supportedFields, field.value, name, checked));
          return (
            <div className="space-y-2">
              {listedFields.map((name) => (
                <Fragment key={name}>
                  <EditableFieldCheckbox
                    name={name}
                    checked={field.value.includes(name)}
                    disabled={isPending}
                    onCheckedChange={toggle(name)}
                  />
                  {name === MAX_BUDGET_FIELD && raiseMaxBudgetSupported && (
                    <EditableFieldCheckbox
                      name={TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION}
                      checked={field.value.includes(TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION)}
                      disabled={isPending || !field.value.includes(MAX_BUDGET_FIELD)}
                      onCheckedChange={toggle(TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION)}
                      className="ml-6"
                      help={<RaiseMaxBudgetHelp />}
                    />
                  )}
                </Fragment>
              ))}
            </div>
          );
        }}
      />
      <div className="flex justify-end">
        <Button type="submit" disabled={isPending || !form.formState.isDirty}>
          {isPending ? "Saving..." : "Save"}
        </Button>
      </div>
    </form>
  );
}

interface EditableFieldCheckboxProps {
  name: string;
  checked: boolean;
  disabled: boolean;
  onCheckedChange: (checked: boolean) => void;
  className?: string;
  help?: ReactNode;
}

function EditableFieldCheckbox({
  name,
  checked,
  disabled,
  onCheckedChange,
  className,
  help,
}: EditableFieldCheckboxProps) {
  const checkboxId = `team-admin-editable-${name}`;
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <label htmlFor={checkboxId} className="flex cursor-pointer items-center gap-2">
        <Checkbox id={checkboxId} checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} />
        <span className="text-sm text-foreground">{teamAdminFieldLabel(name)}</span>
      </label>
      {help}
    </div>
  );
}

function RaiseMaxBudgetHelp() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          type="button"
          aria-label={RAISE_MAX_BUDGET_HELP_LABEL}
          className="inline-flex cursor-help items-center rounded-sm text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <CircleHelp className="size-3.5" />
        </TooltipTrigger>
        <TooltipContent>
          <span>
            {RAISE_MAX_BUDGET_TOOLTIP}{" "}
            <a href={RAISE_MAX_BUDGET_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
              Learn more
            </a>
          </span>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
