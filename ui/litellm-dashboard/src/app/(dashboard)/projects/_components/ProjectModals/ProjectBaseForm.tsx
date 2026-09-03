"use client";

import { useEffect, useState } from "react";
import { useFieldArray, useWatch, type UseFormReturn } from "react-hook-form";
import { ChevronDown, CircleAlert, Minus, Plus } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ALL_TEAM_MODELS, type ProjectFormValues } from "./projectFormSchema";
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

const toOptionalNumber = (raw: string): number | undefined => {
  if (raw.trim() === "") return undefined;
  const parsed = Number(raw);
  return Number.isNaN(parsed) ? undefined : parsed;
};

interface ProjectBaseFormProps {
  form: UseFormReturn<ProjectFormValues>;
  advancedOpen: boolean;
  onAdvancedOpenChange: (open: boolean) => void;
}

export type { ProjectFormValues } from "./projectFormSchema";

export function ProjectBaseForm({ form, advancedOpen, onAdvancedOpenChange }: ProjectBaseFormProps) {
  const { accessToken, userId, userRole } = useAuthorized();
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

  const handleTeamChange = (teamId: string) => {
    const team = teams?.find((t) => t.team_id === teamId) ?? null;
    setSelectedTeam(team);
    form.setValue("models", []);
  };

  const teamOptions = (teams ?? []).map((team) => ({
    value: team.team_id,
    label: team.team_alias || team.team_id,
    sublabel: team.team_id,
  }));

  const modelOptions = [
    { value: ALL_TEAM_MODELS, label: "All Team Models" },
    ...modelsToPick.map((model) => ({ value: model, label: getModelDisplayName(model) })),
  ];
  const modelsPlaceholder = selectedTeam ? "Select models" : "Select a team first";

  return (
    <div className="mt-6">
      <p className="text-xs font-semibold tracking-[0.05em] text-foreground uppercase">Basic Information</p>
      <Separator className="mt-2 mb-4" />

      <FieldGroup>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <FormField control={form.control} name="project_alias" label="Project Name">
            {({ ref, ...field }) => (
              <Input {...field} value={field.value ?? ""} ref={ref} placeholder="e.g. Customer Support Bot" />
            )}
          </FormField>

          <FormField control={form.control} name="team_id" label="Team">
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
                placeholder="Search or select a team"
                allowClear
              />
            )}
          </FormField>
        </div>

        <FormField control={form.control} name="description" label="Description">
          {({ ref, ...field }) => (
            <Textarea
              {...field}
              value={field.value ?? ""}
              ref={ref}
              rows={3}
              placeholder="Describe the purpose of this project"
            />
          )}
        </FormField>

        <FormField
          control={form.control}
          name="models"
          label="Allowed Models (scoped to selected team's models)"
          description={!selectedTeam ? "Select a team first to see available models" : undefined}
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
          <FormField control={form.control} name="max_budget" label="Max Budget (USD)">
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
                  value={value ?? ""}
                  onChange={(event) => onChange(toOptionalNumber(event.target.value))}
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
              <span className="text-sm font-semibold text-foreground">Advanced Settings</span>
            </button>
          }
        />
        <CollapsibleContent className="px-4 pb-4">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-foreground">Block Project</span>
            <FormField control={form.control} name="isBlocked" className="w-auto">
              {({ id, value, onChange, ref: _ref, ...field }) => (
                <Switch {...field} id={id} checked={value} onCheckedChange={onChange} />
              )}
            </FormField>
          </div>

          {isBlocked ? (
            <Alert variant="warning" className="mt-3">
              <CircleAlert />
              <AlertTitle>All API requests using keys under this project will be rejected.</AlertTitle>
            </Alert>
          ) : null}

          <Separator className="my-4" />

          <FormField
            control={form.control}
            name="guardrails"
            label="Guardrails"
            description="Select existing guardrails or enter new ones"
          >
            {({ id, value, onChange }) => (
              <TagsInput
                id={id}
                value={value ?? []}
                onValueChange={onChange}
                options={guardrailsList.map((name) => ({ label: name, value: name }))}
                placeholder="Select or enter guardrails"
              />
            )}
          </FormField>

          <Separator className="my-4" />

          <p className="mb-3 text-sm font-semibold text-foreground">Model-Specific Limits</p>
          {modelLimits.fields.map((field, index) => (
            <div
              key={field.id}
              className="mb-2 grid grid-cols-1 items-start gap-2 sm:grid-cols-2 xl:grid-cols-[minmax(0,2fr)_repeat(4,minmax(0,1fr))_auto]"
            >
              <FormField control={form.control} name={`modelLimits.${index}.model`} label="Model">
                {({ ref, ...control }) => (
                  <Input {...control} value={control.value ?? ""} ref={ref} placeholder="Model name (e.g. gpt-4)" />
                )}
              </FormField>
              <FormField control={form.control} name={`modelLimits.${index}.tpm`} label="TPM Limit">
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder="TPM Limit"
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <FormField control={form.control} name={`modelLimits.${index}.rpm`} label="RPM Limit">
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder="RPM Limit"
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <FormField control={form.control} name={`modelLimits.${index}.itpm`} label="Input TPM Limit">
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder="Input TPM Limit"
                    value={value ?? ""}
                    onChange={(event) => onChange(toOptionalNumber(event.target.value))}
                  />
                )}
              </FormField>
              <FormField control={form.control} name={`modelLimits.${index}.otpm`} label="Output TPM Limit">
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    placeholder="Output TPM Limit"
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
                aria-label={`Remove model limit ${index + 1}`}
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
            Add Model Limit
          </Button>

          <Separator className="my-4" />

          <p className="mb-3 text-sm font-semibold text-foreground">Metadata</p>
          {metadata.fields.map((field, index) => (
            <div key={field.id} className="mb-2 flex items-start gap-2">
              <FormField control={form.control} name={`metadata.${index}.key`}>
                {({ ref, ...control }) => (
                  <Input {...control} value={control.value ?? ""} ref={ref} placeholder="Key" />
                )}
              </FormField>
              <FormField control={form.control} name={`metadata.${index}.value`}>
                {({ ref, ...control }) => (
                  <Input {...control} value={control.value ?? ""} ref={ref} placeholder="Value" />
                )}
              </FormField>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="mt-1 text-destructive"
                onClick={() => metadata.remove(index)}
                aria-label={`Remove metadata pair ${index + 1}`}
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
            Add Key-Value Pair
          </Button>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
