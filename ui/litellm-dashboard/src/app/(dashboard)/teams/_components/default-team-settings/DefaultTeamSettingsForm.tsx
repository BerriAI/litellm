"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as React from "react";
import { type Control } from "react-hook-form";

import { ModelSelect, MODEL_SENTINEL_OPTIONS } from "@/components/ModelSelect/ModelSelect";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { BUDGET_DURATION_OPTIONS, NO_RESET } from "@/components/shared/form/budgetDuration";
import { getPermissionInfo } from "@/components/team/permission_definitions";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";

import { buildBody, settingsToForm, type DefaultTeamParams, type DefaultTeamSettings } from "./mapper";
import { SELECTABLE_PERMISSIONS, type KeyManagementRoute } from "./permissions";
import { defaultTeamSettingsSchema, type DefaultTeamSettingsFormValues } from "./schema";

const MODEL_SENTINEL_LABELS: ReadonlyMap<string, string> = new Map(
  MODEL_SENTINEL_OPTIONS.map(({ value, label }) => [value, label]),
);

const SETTINGS_QUERY_KEY = ["defaultTeamSettings"] as const;

const PERMISSIONS_DESCRIPTION =
  "Default permissions granted to members of newly created teams. /key/info and /key/health are always included.";

const defaultFetchSettings = async (): Promise<DefaultTeamSettings> => {
  const { data } = await fetchClient.GET("/get/default_team_settings");
  if (data === undefined) {
    throw new Error("Failed to load default team settings");
  }
  return data;
};

const defaultUpdateSettings = async (body: DefaultTeamParams): Promise<void> => {
  await fetchClient.PATCH("/update/default_team_settings", { body });
};

type SettingsControl = Control<DefaultTeamSettingsFormValues, unknown, DefaultTeamSettingsFormValues>;

const PermissionRow = ({
  route,
  granted,
  onToggle,
}: {
  route: KeyManagementRoute;
  granted: boolean;
  onToggle: (checked: boolean) => void;
}) => {
  const reactId = React.useId();
  const labelId = `${reactId}-label`;
  const descriptionId = `${reactId}-description`;

  return (
    <label className="flex cursor-pointer items-start gap-3">
      <Checkbox
        aria-labelledby={labelId}
        aria-describedby={descriptionId}
        className="mt-0.5"
        checked={granted}
        onCheckedChange={onToggle}
      />
      <span className="flex flex-col">
        <span id={labelId} className="font-mono text-sm">
          {route}
        </span>
        <span id={descriptionId} className="text-xs text-muted-foreground">
          {getPermissionInfo(route).description}
        </span>
      </span>
    </label>
  );
};

const PermissionsField = ({ control }: { control: SettingsControl }) => (
  <FormField
    control={control}
    name="team_member_permissions"
    label="Team Member Permissions"
    description={PERMISSIONS_DESCRIPTION}
  >
    {({ id, value, onChange }) => (
      <div id={id} className="flex flex-col gap-3">
        {SELECTABLE_PERMISSIONS.map((route) => (
          <PermissionRow
            key={route}
            route={route}
            granted={value.includes(route)}
            onToggle={(checked) =>
              onChange(checked ? [...value, route] : value.filter((existing) => existing !== route))
            }
          />
        ))}
      </div>
    )}
  </FormField>
);

const ViewRow = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div>
    <p className="text-sm font-medium">{label}</p>
    <p className="text-sm text-muted-foreground">{children}</p>
  </div>
);

const SettingsView = ({ values }: { values: DefaultTeamSettingsFormValues }) => {
  const durationValue = values.budget_duration === "" ? NO_RESET : values.budget_duration;
  const durationLabel =
    BUDGET_DURATION_OPTIONS.find((option) => option.value === durationValue)?.label ?? values.budget_duration;

  return (
    <div className="flex flex-col gap-4">
      <ViewRow label="Max Budget (USD)">{values.max_budget === "" ? "Not set" : values.max_budget}</ViewRow>
      <ViewRow label="Reset Budget">{durationLabel}</ViewRow>
      <ViewRow label="TPM Limit">{values.tpm_limit === "" ? "Not set" : values.tpm_limit}</ViewRow>
      <ViewRow label="RPM Limit">{values.rpm_limit === "" ? "Not set" : values.rpm_limit}</ViewRow>
      <ViewRow label="Default Models">
        {values.models.length === 0
          ? "Not set"
          : values.models.map((model) => MODEL_SENTINEL_LABELS.get(model) ?? model).join(", ")}
      </ViewRow>
      <ViewRow label="Team Member Permissions">
        {values.team_member_permissions.length === 0 ? "Not set" : values.team_member_permissions.join(", ")}
      </ViewRow>
    </div>
  );
};

interface SettingsFormProps {
  initialValues: DefaultTeamSettingsFormValues;
  updateSettings: (body: DefaultTeamParams) => Promise<void>;
  onCancel: () => void;
  onSaved: () => void;
}

const SettingsForm = ({ initialValues, updateSettings, onCancel, onSaved }: SettingsFormProps) => {
  const queryClient = useQueryClient();
  const form = useZodForm(defaultTeamSettingsSchema, { defaultValues: initialValues });
  const { isDirty } = form.formState;

  const mutation = useMutation({
    mutationFn: (values: DefaultTeamSettingsFormValues) => updateSettings(buildBody(values)),
    onSuccess: (_result, values) => {
      NotificationsManager.success("Default team settings updated successfully");
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEY });
      form.reset(values);
      onSaved();
    },
    onError: (error: unknown) =>
      NotificationsManager.fromBackend(
        error instanceof Error ? error.message : "Failed to update default team settings",
      ),
  });

  const onSubmit = form.handleSubmit((values) => mutation.mutate(values));

  return (
    <form onSubmit={onSubmit} noValidate>
      <FieldGroup>
        <FormField
          control={form.control}
          name="max_budget"
          label="Max Budget (USD)"
          description="Default maximum budget for new teams"
        >
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={0.01} min={0} />}
        </FormField>

        <FormField
          control={form.control}
          name="budget_duration"
          label="Reset Budget"
          description="How often the default budget resets"
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Select
              items={BUDGET_DURATION_OPTIONS}
              value={value === "" ? NO_RESET : value}
              onValueChange={(selected) => onChange(selected === null || selected === NO_RESET ? "" : selected)}
            >
              <SelectTrigger id={id} className="w-full" aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {BUDGET_DURATION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </FormField>

        <FormField
          control={form.control}
          name="tpm_limit"
          label="TPM Limit"
          description="Default tokens-per-minute limit for new teams"
        >
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
        </FormField>

        <FormField
          control={form.control}
          name="rpm_limit"
          label="RPM Limit"
          description="Default requests-per-minute limit for new teams"
        >
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={1} min={0} />}
        </FormField>

        <FormField
          control={form.control}
          name="models"
          label="Default Models"
          description="Models new teams can access"
        >
          {(field) => (
            <ModelSelect
              value={field.value}
              onChange={field.onChange}
              context="global"
              options={{ includeSpecialOptions: true }}
            />
          )}
        </FormField>

        <PermissionsField control={form.control} />
      </FieldGroup>

      <div className="mt-6 flex items-center justify-end gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={() => {
            form.reset(initialValues);
            onCancel();
          }}
          disabled={mutation.isPending}
        >
          Cancel
        </Button>
        <Button type="submit" disabled={!isDirty || mutation.isPending}>
          {mutation.isPending ? "Saving..." : "Save Changes"}
        </Button>
      </div>
    </form>
  );
};

const SettingsCard = ({ action, children }: { action?: React.ReactNode; children: React.ReactNode }) => (
  <Card>
    <CardHeader>
      <CardTitle>Default Team Settings</CardTitle>
      <CardDescription>
        Applied to every new team automatically created by LiteLLM, such as teams created from SSO groups.
      </CardDescription>
      {action !== undefined && <CardAction>{action}</CardAction>}
    </CardHeader>
    <CardContent>{children}</CardContent>
  </Card>
);

export interface DefaultTeamSettingsFormProps {
  fetchSettings?: () => Promise<DefaultTeamSettings>;
  updateSettings?: (body: DefaultTeamParams) => Promise<void>;
}

export const DefaultTeamSettingsForm = ({
  fetchSettings = defaultFetchSettings,
  updateSettings = defaultUpdateSettings,
}: DefaultTeamSettingsFormProps) => {
  const [isEditing, setIsEditing] = React.useState(false);
  const { data, isPending, isError } = useQuery({ queryKey: SETTINGS_QUERY_KEY, queryFn: fetchSettings });

  const initialValues = React.useMemo(() => (data === undefined ? undefined : settingsToForm(data.values)), [data]);

  if (isPending) {
    return (
      <SettingsCard>
        <Skeleton className="h-64 w-full" />
      </SettingsCard>
    );
  }

  if (isError || initialValues === undefined) {
    return (
      <SettingsCard>
        <p role="alert">Could not load the default team settings.</p>
      </SettingsCard>
    );
  }

  return (
    <SettingsCard
      action={
        isEditing ? undefined : (
          <Button type="button" onClick={() => setIsEditing(true)}>
            Edit Settings
          </Button>
        )
      }
    >
      {isEditing ? (
        <SettingsForm
          initialValues={initialValues}
          updateSettings={updateSettings}
          onCancel={() => setIsEditing(false)}
          onSaved={() => setIsEditing(false)}
        />
      ) : (
        <SettingsView values={initialValues} />
      )}
    </SettingsCard>
  );
};
