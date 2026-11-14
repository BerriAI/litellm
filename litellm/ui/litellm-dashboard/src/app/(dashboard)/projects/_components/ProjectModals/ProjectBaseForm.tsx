"use client";

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useFieldArray, useWatch, type UseFormReturn } from "react-hook-form";
import { ChevronDown, CircleAlert, Minus, Plus } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { ALL_TEAM_MODELS, type ProjectFormValues, type ProjectSubmitValues } from "./projectFormSchema";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "@/components/key_team_helpers/key_list";
import { fetchTeamModels } from "@/components/organisms/create_key_button";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { getGuardrailsList } from "@/components/networking";
import { TagsInput } from "@/app/(dashboard)/guardrails/_components/content_filter/TagsInput";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { teamsUserCanAssign } from "@/utils/roles";

const toOptionalNumber = (raw: string): number | undefined => {
  if (raw.trim() === "") return undefined;
  const parsed = Number(raw);
  return Number.isNaN(parsed) ? undefined : parsed;
};

interface ProjectBaseFormProps {
  form: UseFormReturn<ProjectFormValues, unknown, ProjectSubmitValues>;
  advancedOpen: boolean;
  onAdvancedOpenChange: (open: boolean) => void;
}

export type { ProjectFormValues } from "./projectFormSchema";

export function ProjectBaseForm({ form, advancedOpen, onAdvancedOpenChange }: ProjectBaseFormProps) {
  const { t } = useTranslation();
  const { accessToken, userId, userRole } = useAuthorized();
  const canEditBatchLimit = isProxyAdminRole(userRole ?? "");
  const { data: teams } = useTeams();

  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);

  const modelLimits = useFieldArray({ control: form.control, name: "modelLimits" });
  const metadata = useFieldArray({ control: form.control, name: "metadata" });
  const emptyModelLimit: NonNullable<ProjectFormValues["modelLimits"]>[number] = {
    model: "",
    tpm: undefined,
    rpm: undefined,
    itpm: undefined,
    otpm: undefined,
    batchEnqueuedTokens: undefined,
  };

  const teamIdValue = useWatch({ control: form.control, name: "team_id" });
  const isBlocked = useWatch({ control: form.control, name: "isBlocked" });

  useEffect(() => {
    const fetchGuardrails = async () => {
      if (!accessToken) return;
      try {
        const response = await getGuardrailsList(accessToken);
        const names = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(names);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };
    fetchGuardrails();
  }, [accessToken]);

  useEffect(() => {
    if (teamIdValue && teams) {
      const team = teams.find((t) => t.team_id === teamIdValue) ?? null;
      if (team && team.team_id !== selectedTeam?.team_id) {
        setSelectedTeam(team);
      }
    }
  }, [teamIdValue, teams, selectedTeam?.team_id]);

  useEffect(() => {
    if (userId && userRole && accessToken && selectedTeam) {
      fetchTeamModels(userId, userRole, accessToken, selectedTeam.team_id).then((models) => {
        const allModels = Array.from(new Set([...(selectedTeam.models ?? []), ...models]));
        setModelsToPick(allModels);
      });
    } else {
      setModelsToPick([]);
    }
  }, [selectedTeam, accessToken, userId, userRole]);

  const handleTeamChange = (teamId: string | null) => {
    const team = teams?.find((t) => t.team_id === teamId) ?? null;
    setSelectedTeam(team);
    form.setValue("models", []);
  };

  const teamOptions = (teamsUserCanAssign(teams ?? null, userRole, userId) ?? []).map((team) => ({
    value: team.team_id,
    label: team.team_alias || team.team_id,
    sublabel: team.team_id,
  }));

  const modelOptions = [
    { value: ALL_TEAM_MODELS, label: t("projects.form.allTeamModels") },
    ...modelsToPick.map((model) => ({ value: model, label: getModelDisplayName(model) })),
  ];
  const modelsPlaceholder = selectedTeam ? t("projects.form.selectModels") : t("projects.form.selectTeamFirst");

  return (
    <div className="mt-6">
      <p className="text-xs font-semibold tracking-[0.05em] text-foreground uppercase">
        {t("projects.form.basicInformation")}
      </p>
      <Separator className="mt-2 mb-4" />

      <FieldGroup>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <FormField control={form.control} name="project_alias" label={t("projects.form.projectName")}>
            {({ ref, ...field }) => (
              <Input
                {...field}
                value={field.value ?? ""}
                ref={ref}
                placeholder={t("projects.form.projectNamePlaceholder")}
              />
            )}
          </FormField>

          <FormField control={form.control} name="team_id" label={t("projects.form.team")}>
            {({ id, value, onChange, ref: _ref, ...field }) => (
              <SearchSelect
                {...field}
                inputId={id}
                options={teamOptions}
                value={value}
                onValueChange={(next) => {
                  onChange(next);
                  handleTeamChange(next);
                }}
                placeholder={t("projects.form.teamPlaceholder")}
                allowClear
              />
            )}
          </FormField>
        </div>

        <FormField control={form.control} name="description" label={t("projects.form.description")}>
          {({ ref, ...field }) => (
            <Textarea
              {...field}
              value={field.value ?? ""}
              ref={ref}
              rows={3}
              placeholder={t("projects.form.descriptionPlaceholder")}
            />
          )}
        </FormField>

        <FormField
          control={form.control}
          name="models"
          label={t("projects.form.allowedModels")}
          description={!selectedTeam ? t("projects.form.allowedModelsHint") : undefined}
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Select
              multiple
              items={modelOptions}
              value={value}
              onValueChange={(next: string[]) => onChange(next.includes(ALL_TEAM_MODELS) ? [ALL_TEAM_MODELS] : next)}
              disabled={!selectedTeam}
            >
              <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} className="w-full">
                <SelectValue placeholder={modelsPlaceholder}>
                  {(selected: string[]) =>
                    selected.length === 0
                      ? modelsPlaceholder
                      : modelOptions
                          .filter((option) => selected.includes(option.value))
                          .map((option) => option.label)
                          .join(", ")
                  }
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {modelOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value} title={option.label}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </FormField>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <FormField control={form.control} name="max_budget" label={t("projects.form.maxBudget")}>
            {({ ref, value, onChange, ...field }) => (
              <InputGroup>
                <InputGroupAddon>
                  <InputGroupText>$</InputGroupText>
                </InputGroupAddon>
                <InputGroupInput
                  {...field}
                  ref={ref}
                  type="number"
                  min={0}
                  placeholder="0.00"
                  value={Number.isNaN(value) ? "" : value ?? ""}
                  onInput={(event) => {
                    if (event.currentTarget.validity.badInput || Number.isNaN(value)) {
                      onChange(
                        event.currentTarget.validity.badInput
                          ? Number.NaN
                          : toOptionalNumber(event.currentTarget.value) ?? null,
                      );
                    }
                  }}
                  onChange={(event) =>
                    onChange(event.target.validity.badInput ? Number.NaN : toOptionalNumber(event.target.value) ?? null)
                  }
                />
              </InputGroup>
            )}
          </FormField>
        </div>
      </FieldGroup>

      <Collapsible
        open={advancedOpen}
        onOpenChange={onAdvancedOpenChange}
        className="mt-6 rounded-lg border border-border bg-muted"
      >
        <CollapsibleTrigger
          render={
            <button type="button" className="flex w-full items-center gap-2 px-4 py-3 text-left">
              <ChevronDown
                className={`size-4 text-muted-foreground transition-transform ${advancedOpen ? "" : "-rotate-90"}`}
              />
              <span className="text-sm font-semibold text-foreground">{t("projects.form.advancedSettings")}</span>
            </button>
          }
        />
        <CollapsibleContent className="px-4 pb-4">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-foreground">{t("projects.form.blockProject")}</span>
            <FormField control={form.control} name="isBlocked" className="w-auto">
              {({ id, value, onChange, ref: _ref, ...field }) => (
                <Switch {...field} id={id} checked={value} onCheckedChange={onChange} />
              )}
            </FormField>
          </div>

          {isBlocked ? (
            <Alert variant="warning" className="mt-3">
              <CircleAlert />
              <AlertTitle>{t("projects.form.blockedWarning")}</AlertTitle>
            </Alert>
          ) : null}

          <Separator className="my-4" />

          <FormField
            control={form.control}
            name="guardrails"
            label={t("projects.form.guardrails")}
            description={t("projects.form.guardrailsHint")}
          >
            {({ id, value, onChange }) => (
              <TagsInput
                id={id}
                value={value ?? []}
                onValueChange={onChange}
                options={guardrailsList.map((name) => ({ label: name, value: name }))}
                placeholder={t("projects.form.guardrailsPlaceholder")}
              />
            )}
          </FormField>

          <Separator className="my-4" />

          <p className="mb-3 text-sm font-semibold text-foreground">{t("projects.form.modelLimits.title")}</p>
          {modelLimits.fields.map((field, index) => (
            <div
              key={field.id}
              className="mb-2 grid grid-cols-1 items-start gap-2 sm:grid-cols-2 xl:grid-cols-[minmax(0,2fr)_repeat(5,minmax(0,1fr))_auto]"
            >
              <FormField
                control={form.control}
                name={`modelLimits.${index}.model`}
                label={t("projects.form.modelLimits.model")}
              >
                {({ ref, ...control }) => (
                  <Input
                    {...control}
                    value={control.value ?? ""}
                    ref={ref}
                    placeholder={t("projects.form.modelLimits.modelPlaceholder")}
                  />
                )}
              </FormField>
              <FormField
                control={form.control}
                name={`modelLimits.${index}.tpm`}
                label={t("projects.form.modelLimits.tpm")}
              >
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder={t("projects.form.modelLimits.tpm")}
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <FormField
                control={form.control}
                name={`modelLimits.${index}.rpm`}
                label={t("projects.form.modelLimits.rpm")}
              >
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder={t("projects.form.modelLimits.rpm")}
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <FormField
                control={form.control}
                name={`modelLimits.${index}.itpm`}
                label={t("projects.form.modelLimits.itpm")}
              >
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder={t("projects.form.modelLimits.itpm")}
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <FormField
                control={form.control}
                name={`modelLimits.${index}.otpm`}
                label={t("projects.form.modelLimits.otpm")}
              >
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder={t("projects.form.modelLimits.otpm")}
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <FormField
                control={form.control}
                name={`modelLimits.${index}.batchEnqueuedTokens`}
                label={t("projects.form.modelLimits.batchEnqueuedTokens")}
              >
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={1}
                    step={1}
                    disabled={!canEditBatchLimit}
                    placeholder={t("projects.form.modelLimits.batchEnqueuedTokens")}
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="mt-1 text-destructive"
                onClick={() => modelLimits.remove(index)}
                aria-label={t("projects.form.modelLimits.remove", { number: index + 1 })}
              >
                <Minus />
              </Button>
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            className="w-full border-dashed"
            onClick={() => modelLimits.append(emptyModelLimit)}
          >
            <Plus />
            {t("projects.form.modelLimits.add")}
          </Button>

          <Separator className="my-4" />

          <p className="mb-3 text-sm font-semibold text-foreground">{t("projects.form.metadata.title")}</p>
          {metadata.fields.map((field, index) => (
            <div key={field.id} className="mb-2 flex items-start gap-2">
              <FormField control={form.control} name={`metadata.${index}.key`}>
                {({ ref, ...control }) => (
                  <Input
                    {...control}
                    value={control.value ?? ""}
                    ref={ref}
                    placeholder={t("projects.form.metadata.key")}
                  />
                )}
              </FormField>
              <FormField control={form.control} name={`metadata.${index}.value`}>
                {({ ref, ...control }) => (
                  <Input
                    {...control}
                    value={control.value ?? ""}
                    ref={ref}
                    placeholder={t("projects.form.metadata.value")}
                  />
                )}
              </FormField>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="mt-1 text-destructive"
                onClick={() => metadata.remove(index)}
                aria-label={t("projects.form.metadata.remove", { number: index + 1 })}
              >
                <Minus />
              </Button>
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            className="w-full border-dashed"
            onClick={() => metadata.append({ key: "", value: "" })}
          >
            <Plus />
            {t("projects.form.metadata.add")}
          </Button>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
