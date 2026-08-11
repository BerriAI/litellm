"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as React from "react";
import { useTranslation } from "react-i18next";
import { useFieldArray, type Control } from "react-hook-form";

import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { ModelSelect, MODEL_SENTINEL_OPTIONS } from "@/components/ModelSelect/ModelSelect";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";

import { buildBody, settingsToForm, type DefaultInternalUserParams, type InternalUserSettings } from "./mapper";
import { createDefaultUserSettingsSchema, EMPTY_TEAM_ROW, type DefaultUserSettingsFormValues } from "./schema";

const NO_RESET = "never";

const BUDGET_DURATION_VALUES = [NO_RESET, "1h", "24h", "7d", "30d"] as const;

const TEAM_ROLE_VALUES = ["user", "admin"] as const;

const TEAMS_PAGE_SIZE = 50;

const SETTINGS_QUERY_KEY = ["internalUserSettings"] as const;

const defaultFetchSettings = async (): Promise<InternalUserSettings> => {
  const { data } = await fetchClient.GET("/get/internal_user_settings");
  if (data === undefined) {
    throw new Error("Failed to load default user settings");
  }
  return data;
};

const defaultUpdateSettings = async (body: DefaultInternalUserParams): Promise<void> => {
  await fetchClient.PATCH("/update/internal_user_settings", { body });
};

interface RoleOption {
  value: string;
  label: string;
  description: string;
}

type SettingsControl = Control<DefaultUserSettingsFormValues, unknown, DefaultUserSettingsFormValues>;

const TeamPickerField = ({ control, index }: { control: SettingsControl; index: number }) => {
  const { t } = useTranslation("gateway");
  const [search, setSearch] = React.useState("");
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteTeams(
    TEAMS_PAGE_SIZE,
    search === "" ? undefined : search,
  );

  const options = React.useMemo<SearchSelectOption[]>(
    () =>
      (data?.pages ?? []).flatMap((page) =>
        page.teams.map((team) => ({
          label: team.team_alias || team.team_id,
          value: team.team_id,
          sublabel: team.team_id,
        })),
      ),
    [data],
  );

  return (
    <FormField control={control} name={`teams.${index}.team_id`} label={t("users.fields.team")}>
      {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
        <PaginatedSearchSelect
          options={options}
          value={value}
          onValueChange={onChange}
          onSearchChange={setSearch}
          onLoadMore={() => void fetchNextPage()}
          hasNextPage={hasNextPage}
          isLoading={isLoading}
          isFetchingNextPage={isFetchingNextPage}
          placeholder={t("users.defaultSettings.searchTeam")}
          emptyText={t("users.defaultSettings.noTeams")}
          inputId={id}
          aria-invalid={ariaInvalid}
          aria-describedby={ariaDescribedBy}
        />
      )}
    </FormField>
  );
};

const TeamsField = ({ control }: { control: SettingsControl }) => {
  const { t } = useTranslation("gateway");
  const { fields, append, remove } = useFieldArray({ control, name: "teams" });
  const teamRoleOptions = React.useMemo(
    () => [
      { value: TEAM_ROLE_VALUES[0], label: t("users.defaultSettings.roleUser") },
      { value: TEAM_ROLE_VALUES[1], label: t("users.defaultSettings.roleAdmin") },
    ],
    [t],
  );

  return (
    <div className="flex w-full flex-col gap-3">
      <div>
        <p className="text-sm font-medium">{t("users.defaultSettings.defaultTeams")}</p>
        <p className="text-sm text-muted-foreground">{t("users.defaultSettings.teamsDescription")}</p>
      </div>

      {fields.map((field, index) => (
        <div key={field.id} className="rounded-lg border border-border p-4">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-medium">{t("users.defaultSettings.teamNumber", { number: index + 1 })}</p>
            <Button type="button" variant="destructive" size="sm" onClick={() => remove(index)}>
              {t("users.defaultSettings.remove")}
            </Button>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <TeamPickerField control={control} index={index} />

            <FormField
              control={control}
              name={`teams.${index}.max_budget_in_team`}
              label={t("users.defaultSettings.teamBudget")}
            >
              {({ ref, ...budgetField }) => (
                <Input
                  {...budgetField}
                  ref={ref}
                  type="number"
                  step="any"
                  min={0}
                  placeholder={t("users.defaultSettings.optional")}
                />
              )}
            </FormField>

            <FormField control={control} name={`teams.${index}.user_role`} label={t("users.defaultSettings.teamRole")}>
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                <Select
                  items={teamRoleOptions}
                  value={value}
                  onValueChange={(selected) => onChange(selected ?? "user")}
                >
                  <SelectTrigger
                    id={id}
                    className="w-full"
                    aria-invalid={ariaInvalid}
                    aria-describedby={ariaDescribedBy}
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {teamRoleOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>
          </div>
        </div>
      ))}

      <Button type="button" variant="outline" onClick={() => append(EMPTY_TEAM_ROW)}>
        {t("users.defaultSettings.addTeam")}
      </Button>
    </div>
  );
};

const ViewRow = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div>
    <p className="text-sm font-medium">{label}</p>
    <p className="text-sm text-muted-foreground">{children}</p>
  </div>
);

interface SettingsViewProps {
  values: DefaultUserSettingsFormValues;
  roleOptions: readonly RoleOption[];
}

const SettingsView = ({ values, roleOptions }: SettingsViewProps) => {
  const { t } = useTranslation("gateway");
  const roleLabel = roleOptions.find((option) => option.value === values.user_role)?.label ?? values.user_role;
  const durationValue = values.budget_duration === "" ? NO_RESET : values.budget_duration;
  const durationLabels: Record<string, string> = {
    [NO_RESET]: t("users.defaultSettings.noReset"),
    "1h": t("users.defaultSettings.hourly"),
    "24h": t("users.defaultSettings.daily"),
    "7d": t("users.defaultSettings.weekly"),
    "30d": t("users.defaultSettings.monthly"),
  };
  const durationLabel = durationLabels[durationValue] ?? values.budget_duration;
  const modelSentinelLabels: ReadonlyMap<string, string> = new Map(
    MODEL_SENTINEL_OPTIONS.map(({ value, label }) => [
      value,
      value === "all-proxy-models"
        ? t("users.invite.allProxyModels")
        : value === "no-default-models"
          ? t("users.invite.noDefaultModels")
          : label,
    ]),
  );

  return (
    <div className="flex flex-col gap-4">
      <ViewRow label={t("users.defaultSettings.defaultRole")}>
        {roleLabel === "" ? t("users.defaultSettings.notSet") : roleLabel}
      </ViewRow>
      <ViewRow label={t("users.defaultSettings.maxBudget")}>
        {values.max_budget === "" ? t("users.defaultSettings.notSet") : values.max_budget}
      </ViewRow>
      <ViewRow label={t("users.defaultSettings.resetBudget")}>{durationLabel}</ViewRow>
      <ViewRow label={t("users.defaultSettings.defaultModels")}>
        {values.models.length === 0
          ? t("users.defaultSettings.notSet")
          : values.models.map((model) => modelSentinelLabels.get(model) ?? model).join(", ")}
      </ViewRow>
      <div>
        <p className="text-sm font-medium">{t("users.defaultSettings.defaultTeams")}</p>
        {values.teams.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("users.defaultSettings.none")}</p>
        ) : (
          values.teams.map((team) => (
            <p key={team.team_id} className="text-sm text-muted-foreground">
              {team.team_id}
              {team.max_budget_in_team !== "" && (
                <>
                  {" "}
                  · ${team.max_budget_in_team} {t("users.defaultSettings.maxBudgetSuffix")}
                </>
              )}
              <> · {team.user_role}</>
            </p>
          ))
        )}
      </div>
    </div>
  );
};

interface SettingsFormProps {
  initialValues: DefaultUserSettingsFormValues;
  roleOptions: readonly RoleOption[];
  updateSettings: (body: DefaultInternalUserParams) => Promise<void>;
  onCancel: () => void;
  onSaved: () => void;
}

const SettingsForm = ({ initialValues, roleOptions, updateSettings, onCancel, onSaved }: SettingsFormProps) => {
  const { t } = useTranslation("gateway");
  const queryClient = useQueryClient();
  const validationSchema = React.useMemo(
    () =>
      createDefaultUserSettingsSchema({
        nonNegative: t("users.defaultSettings.validation.nonNegative"),
        selectTeam: t("users.defaultSettings.validation.selectTeam"),
        duplicateTeam: t("users.defaultSettings.validation.duplicateTeam"),
      }),
    [t],
  );
  const budgetDurationOptions = React.useMemo(
    () => [
      { value: BUDGET_DURATION_VALUES[0], label: t("users.defaultSettings.noReset") },
      { value: BUDGET_DURATION_VALUES[1], label: t("users.defaultSettings.hourly") },
      { value: BUDGET_DURATION_VALUES[2], label: t("users.defaultSettings.daily") },
      { value: BUDGET_DURATION_VALUES[3], label: t("users.defaultSettings.weekly") },
      { value: BUDGET_DURATION_VALUES[4], label: t("users.defaultSettings.monthly") },
    ],
    [t],
  );
  const form = useZodForm(validationSchema, { defaultValues: initialValues });
  const { isDirty } = form.formState;

  const mutation = useMutation({
    mutationFn: (values: DefaultUserSettingsFormValues) => updateSettings(buildBody(values)),
    onSuccess: (_result, values) => {
      NotificationsManager.success(t("users.defaultSettings.updated"));
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEY });
      form.reset(values);
      onSaved();
    },
    onError: (error: unknown) =>
      NotificationsManager.fromBackend(
        error instanceof Error ? error.message : t("users.defaultSettings.updateFailed"),
      ),
  });

  const onSubmit = form.handleSubmit((values) => mutation.mutate(values));

  return (
    <form onSubmit={onSubmit} noValidate>
      <FieldGroup>
        <FormField
          control={form.control}
          name="user_role"
          label={t("users.defaultSettings.defaultRole")}
          description={t("users.defaultSettings.roleDescription")}
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Select
              items={roleOptions}
              value={value === "" ? null : value}
              onValueChange={(selected) => onChange(selected ?? "")}
            >
              <SelectTrigger id={id} className="w-full" aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                <SelectValue placeholder={t("users.defaultSettings.notSet")} />
              </SelectTrigger>
              <SelectContent>
                {roleOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    <span>{option.label}</span>
                    {option.description !== "" && (
                      <span className="text-xs text-muted-foreground">{option.description}</span>
                    )}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </FormField>

        <FormField
          control={form.control}
          name="max_budget"
          label={t("users.defaultSettings.maxBudget")}
          description={t("users.defaultSettings.maxBudgetDescription")}
        >
          {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step="any" min={0} />}
        </FormField>

        <FormField
          control={form.control}
          name="budget_duration"
          label={t("users.defaultSettings.resetBudget")}
          description={t("users.defaultSettings.resetDescription")}
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Select
              items={budgetDurationOptions}
              value={value === "" ? NO_RESET : value}
              onValueChange={(selected) => onChange(selected === null || selected === NO_RESET ? "" : selected)}
            >
              <SelectTrigger id={id} className="w-full" aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {budgetDurationOptions.map((option) => (
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
          name="models"
          label={t("users.defaultSettings.defaultModels")}
          description={t("users.defaultSettings.modelsDescription")}
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

        <TeamsField control={form.control} />
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
          {t("users.defaultSettings.cancel")}
        </Button>
        <Button type="submit" disabled={!isDirty || mutation.isPending}>
          {mutation.isPending ? t("users.defaultSettings.saving") : t("users.defaultSettings.save")}
        </Button>
      </div>
    </form>
  );
};

const SettingsCard = ({ action, children }: { action?: React.ReactNode; children: React.ReactNode }) => {
  const { t } = useTranslation("gateway");

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("users.defaultSettings.title")}</CardTitle>
        <CardDescription>{t("users.defaultSettings.description")}</CardDescription>
        {action !== undefined && <CardAction>{action}</CardAction>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
};

export interface DefaultUserSettingsFormProps {
  possibleUIRoles?: Record<string, Record<string, string>> | null;
  fetchSettings?: () => Promise<InternalUserSettings>;
  updateSettings?: (body: DefaultInternalUserParams) => Promise<void>;
}

export const DefaultUserSettingsForm = ({
  possibleUIRoles,
  fetchSettings = defaultFetchSettings,
  updateSettings = defaultUpdateSettings,
}: DefaultUserSettingsFormProps) => {
  const { t } = useTranslation("gateway");
  const [isEditing, setIsEditing] = React.useState(false);
  const { data, isPending, isError } = useQuery({ queryKey: SETTINGS_QUERY_KEY, queryFn: fetchSettings });

  const roleOptions = React.useMemo<RoleOption[]>(
    () =>
      Object.entries(possibleUIRoles ?? {})
        .filter(([role]) => role.includes("internal_user"))
        .map(([role, meta]) => ({ value: role, label: meta.ui_label || role, description: meta.description ?? "" })),
    [possibleUIRoles],
  );

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
        <p role="alert">{t("users.defaultSettings.loadFailed")}</p>
      </SettingsCard>
    );
  }

  return (
    <SettingsCard
      action={
        isEditing ? undefined : (
          <Button type="button" onClick={() => setIsEditing(true)}>
            {t("users.defaultSettings.edit")}
          </Button>
        )
      }
    >
      {isEditing ? (
        <SettingsForm
          initialValues={initialValues}
          roleOptions={roleOptions}
          updateSettings={updateSettings}
          onCancel={() => setIsEditing(false)}
          onSaved={() => setIsEditing(false)}
        />
      ) : (
        <SettingsView values={initialValues} roleOptions={roleOptions} />
      )}
    </SettingsCard>
  );
};
