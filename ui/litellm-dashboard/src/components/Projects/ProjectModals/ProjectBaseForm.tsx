import { useEffect, useMemo, useState } from "react";
import {
  Controller,
  useFieldArray,
  useFormContext,
  useWatch,
} from "react-hook-form";
import { ChevronDown, Minus, Plus, X } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "../../key_team_helpers/key_list";
import { fetchTeamModels } from "../../organisms/create_key_button";
import { getModelDisplayName } from "../../key_team_helpers/fetch_available_models_team_key";
import { getGuardrailsList } from "@/components/networking";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

export interface ProjectFormValues {
  project_alias: string;
  team_id: string;
  description?: string;
  models: string[];
  max_budget?: number;
  isBlocked: boolean;
  guardrails?: string[];
  modelLimits?: { model: string; tpm?: number; rpm?: number }[];
  metadata?: { key: string; value: string }[];
}

export const emptyProjectFormValues: ProjectFormValues = {
  project_alias: "",
  team_id: "",
  description: "",
  models: [],
  max_budget: undefined,
  isBlocked: false,
  guardrails: undefined,
  modelLimits: undefined,
  metadata: undefined,
};

const ALL_TEAM_MODELS = "all-team-models";

/**
 * Multi-select rendered with shadcn Select + chip list below. Accepts a fixed
 * list of options. Mirrors the canonical pattern in
 * `AccessGroupBaseForm.tsx`.
 */
function MultiSelect({
  value,
  onChange,
  options,
  placeholder,
  emptyText,
  disabled,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  emptyText: string;
  disabled?: boolean;
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
          if (!v) return;
          if (v === ALL_TEAM_MODELS) {
            onChange([ALL_TEAM_MODELS]);
            return;
          }
          onChange([...selected.filter((s) => s !== ALL_TEAM_MODELS), v]);
        }}
        disabled={disabled}
      >
        <SelectTrigger aria-label={placeholder}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {emptyText}
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
                <button
                  type="button"
                  onClick={() =>
                    onChange(selected.filter((s) => s !== v))
                  }
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

/**
 * Free-form tag input. Renders a combobox with known options and accepts
 * new entries on Enter / comma. Used for Guardrails.
 */
function TagsSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
}) {
  const [draft, setDraft] = useState("");
  const selected = value ?? [];
  const remaining = options.filter((o) => !selected.includes(o.value));

  const addDraft = () => {
    const v = draft.trim();
    if (!v) return;
    if (selected.includes(v)) {
      setDraft("");
      return;
    }
    onChange([...selected, v]);
    setDraft("");
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          placeholder={placeholder}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              addDraft();
            }
          }}
          onBlur={() => {
            if (draft.trim()) addDraft();
          }}
        />
        {remaining.length > 0 && (
          <Select
            value=""
            onValueChange={(v) => {
              if (v && !selected.includes(v)) {
                onChange([...selected, v]);
              }
            }}
          >
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Pick existing…" />
            </SelectTrigger>
            <SelectContent>
              {remaining.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => (
            <Badge key={v} variant="secondary" className="flex items-center gap-1">
              {v}
              <button
                type="button"
                onClick={() => onChange(selected.filter((s) => s !== v))}
                className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                aria-label={`Remove ${v}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
      {children}
    </span>
  );
}

/**
 * Fields-only block — expected to be rendered inside a parent form managed
 * by `react-hook-form` via a `FormProvider`. See `CreateProjectModal` /
 * `EditProjectModal` for the integration.
 */
export function ProjectBaseForm() {
  const {
    control,
    register,
    setValue,
    formState: { errors },
  } = useFormContext<ProjectFormValues>();

  const { accessToken, userId, userRole } = useAuthorized();
  const { data: teams } = useTeams();

  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const teamIdValue = useWatch({ control, name: "team_id" });
  const isBlockedValue = useWatch({ control, name: "isBlocked" });

  const {
    fields: modelLimitFields,
    append: appendModelLimit,
    remove: removeModelLimit,
  } = useFieldArray({ control, name: "modelLimits" as const });
  const {
    fields: metadataFields,
    append: appendMetadata,
    remove: removeMetadata,
  } = useFieldArray({ control, name: "metadata" as const });

  useEffect(() => {
    const fetchGuardrails = async () => {
      if (!accessToken) return;
      try {
        const response = await getGuardrailsList(accessToken);
        const names = response.guardrails.map(
          (g: { guardrail_name: string }) => g.guardrail_name,
        );
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
    } else if (!teamIdValue && selectedTeam) {
      setSelectedTeam(null);
    }
  }, [teamIdValue, teams, selectedTeam]);

  useEffect(() => {
    if (userId && userRole && accessToken && selectedTeam) {
      fetchTeamModels(userId, userRole, accessToken, selectedTeam.team_id).then(
        (models) => {
          const allModels = Array.from(
            new Set([...(selectedTeam.models ?? []), ...models]),
          );
          setModelsToPick(allModels);
        },
      );
    } else {
      setModelsToPick([]);
    }
  }, [selectedTeam, accessToken, userId, userRole]);

  const teamOptions = (teams ?? []).map((team) => ({
    label: team.team_alias || team.team_id,
    value: team.team_id,
    subLabel: team.team_id,
  }));

  const modelOptions = [
    { label: "All Team Models", value: ALL_TEAM_MODELS },
    ...modelsToPick.map((m) => ({
      label: getModelDisplayName(m),
      value: m,
    })),
  ];

  const guardrailOptions = guardrailsList.map((name) => ({
    label: name,
    value: name,
  }));

  return (
    <div className="space-y-4 mt-2">
      {/* Basic Info */}
      <SectionHeader>Basic Information</SectionHeader>
      <Separator className="mt-1 mb-3" />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="project_alias">
            Project Name <span className="text-destructive">*</span>
          </Label>
          <Input
            id="project_alias"
            placeholder="e.g. Customer Support Bot"
            aria-invalid={!!errors.project_alias}
            {...register("project_alias", {
              required: "Please enter a project name",
            })}
          />
          {errors.project_alias && (
            <p className="text-sm text-destructive">
              {errors.project_alias.message as string}
            </p>
          )}
        </div>
        <div className="space-y-2">
          <Label htmlFor="team_id">
            Team <span className="text-destructive">*</span>
          </Label>
          <Controller
            control={control}
            name="team_id"
            rules={{ required: "Please select a team" }}
            render={({ field }) => (
              <Select
                value={field.value || ""}
                onValueChange={(value) => {
                  field.onChange(value);
                  setValue("models", [], { shouldDirty: true });
                  const team =
                    teams?.find((t) => t.team_id === value) ?? null;
                  setSelectedTeam(team);
                }}
              >
                <SelectTrigger id="team_id" aria-invalid={!!errors.team_id}>
                  <SelectValue placeholder="Search or select a team" />
                </SelectTrigger>
                <SelectContent>
                  {teamOptions.length === 0 ? (
                    <div className="py-2 px-3 text-sm text-muted-foreground">
                      No teams available
                    </div>
                  ) : (
                    teamOptions.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        <span className="font-medium">{opt.label}</span>{" "}
                        <span className="text-muted-foreground">
                          ({opt.subLabel})
                        </span>
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            )}
          />
          {errors.team_id && (
            <p className="text-sm text-destructive">
              {errors.team_id.message as string}
            </p>
          )}
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          placeholder="Describe the purpose of this project"
          rows={3}
          {...register("description")}
        />
      </div>

      <div className="space-y-2">
        <Label>
          Allowed Models (scoped to selected team&apos;s models)
        </Label>
        <Controller
          control={control}
          name="models"
          render={({ field }) => (
            <MultiSelect
              value={field.value ?? []}
              onChange={field.onChange}
              options={modelOptions}
              placeholder={
                selectedTeam ? "Select models" : "Select a team first"
              }
              emptyText={
                selectedTeam
                  ? "No more models available"
                  : "Select a team first"
              }
              disabled={!selectedTeam}
            />
          )}
        />
        {!selectedTeam && (
          <p className="text-sm text-muted-foreground">
            Select a team first to see available models
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="max_budget">Max Budget (USD)</Label>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
              $
            </span>
            <Input
              id="max_budget"
              type="number"
              min={0}
              step="0.01"
              placeholder="0.00"
              className="pl-6"
              {...register("max_budget", {
                setValueAs: (v) =>
                  v === "" || v === null || v === undefined
                    ? undefined
                    : Number(v),
                min: { value: 0, message: "Must be \u2265 0" },
              })}
            />
          </div>
          {errors.max_budget && (
            <p className="text-sm text-destructive">
              {errors.max_budget.message as string}
            </p>
          )}
        </div>
      </div>

      {/* Advanced Settings */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <div className="rounded-md border border-border bg-muted/40">
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="w-full flex justify-between items-center px-4 py-3 text-sm font-semibold text-foreground"
            >
              Advanced Settings
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform",
                  advancedOpen && "rotate-180",
                )}
              />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent className="px-4 pb-4 space-y-4">
            <div className="flex items-center gap-3">
              <Label htmlFor="isBlocked" className="font-semibold">
                Block Project
              </Label>
              <Controller
                control={control}
                name="isBlocked"
                render={({ field }) => (
                  <Switch
                    id="isBlocked"
                    checked={!!field.value}
                    onCheckedChange={field.onChange}
                  />
                )}
              />
            </div>
            {isBlockedValue && (
              <Alert variant="destructive">
                <AlertDescription>
                  All API requests using keys under this project will be
                  rejected.
                </AlertDescription>
              </Alert>
            )}

            <Separator />

            <div className="space-y-2">
              <Label>Guardrails</Label>
              <Controller
                control={control}
                name="guardrails"
                render={({ field }) => (
                  <TagsSelect
                    value={field.value ?? []}
                    onChange={field.onChange}
                    options={guardrailOptions}
                    placeholder="Enter a guardrail and press Enter"
                  />
                )}
              />
              <p className="text-xs text-muted-foreground">
                Select existing guardrails or enter new ones
              </p>
            </div>

            <Separator />

            <div className="space-y-2">
              <Label className="font-semibold">Model-Specific Limits</Label>
              <div className="space-y-2">
                {modelLimitFields.map((field, index) => {
                  const rowErrors = errors.modelLimits?.[index];
                  return (
                    <div
                      key={field.id}
                      className="flex flex-wrap items-start gap-2"
                    >
                      <div className="flex-1 min-w-[180px]">
                        <Input
                          placeholder="Model name (e.g. gpt-4)"
                          aria-invalid={!!rowErrors?.model}
                          {...register(
                            `modelLimits.${index}.model` as const,
                            {
                              required: "Missing model",
                              validate: (value, formValues) => {
                                if (!value) return true;
                                const dupes = (
                                  formValues.modelLimits ?? []
                                ).filter(
                                  (e: { model?: string }) =>
                                    e?.model === value,
                                );
                                return dupes.length > 1
                                  ? "Duplicate model"
                                  : true;
                              },
                            },
                          )}
                        />
                        {rowErrors?.model && (
                          <p className="text-sm text-destructive mt-1">
                            {rowErrors.model.message as string}
                          </p>
                        )}
                      </div>
                      <Input
                        className="w-32"
                        type="number"
                        min={0}
                        placeholder="TPM Limit"
                        {...register(`modelLimits.${index}.tpm` as const, {
                          setValueAs: (v) =>
                            v === "" || v === null || v === undefined
                              ? undefined
                              : Number(v),
                        })}
                      />
                      <Input
                        className="w-32"
                        type="number"
                        min={0}
                        placeholder="RPM Limit"
                        {...register(`modelLimits.${index}.rpm` as const, {
                          setValueAs: (v) =>
                            v === "" || v === null || v === undefined
                              ? undefined
                              : Number(v),
                        })}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => removeModelLimit(index)}
                        aria-label="Remove model limit"
                        className="text-destructive"
                      >
                        <Minus className="h-4 w-4" />
                      </Button>
                    </div>
                  );
                })}
                <Button
                  type="button"
                  variant="outline"
                  onClick={() =>
                    appendModelLimit({
                      model: "",
                      tpm: undefined,
                      rpm: undefined,
                    })
                  }
                  className="w-full border-dashed"
                >
                  <Plus className="h-4 w-4" />
                  Add Model Limit
                </Button>
              </div>
            </div>

            <Separator />

            <div className="space-y-2">
              <Label className="font-semibold">Metadata</Label>
              <div className="space-y-2">
                {metadataFields.map((field, index) => {
                  const rowErrors = errors.metadata?.[index];
                  return (
                    <div
                      key={field.id}
                      className="flex flex-wrap items-start gap-2"
                    >
                      <div className="flex-1 min-w-[160px]">
                        <Input
                          placeholder="Key"
                          aria-invalid={!!rowErrors?.key}
                          {...register(`metadata.${index}.key` as const, {
                            required: "Missing key",
                            validate: (value, formValues) => {
                              if (!value) return true;
                              const dupes = (
                                formValues.metadata ?? []
                              ).filter(
                                (e: { key?: string }) => e?.key === value,
                              );
                              return dupes.length > 1
                                ? "Duplicate key"
                                : true;
                            },
                          })}
                        />
                        {rowErrors?.key && (
                          <p className="text-sm text-destructive mt-1">
                            {rowErrors.key.message as string}
                          </p>
                        )}
                      </div>
                      <div className="flex-1 min-w-[160px]">
                        <Input
                          placeholder="Value"
                          aria-invalid={!!rowErrors?.value}
                          {...register(
                            `metadata.${index}.value` as const,
                            { required: "Missing value" },
                          )}
                        />
                        {rowErrors?.value && (
                          <p className="text-sm text-destructive mt-1">
                            {rowErrors.value.message as string}
                          </p>
                        )}
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => removeMetadata(index)}
                        aria-label="Remove metadata entry"
                        className="text-destructive"
                      >
                        <Minus className="h-4 w-4" />
                      </Button>
                    </div>
                  );
                })}
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => appendMetadata({ key: "", value: "" })}
                  className="w-full border-dashed"
                >
                  <Plus className="h-4 w-4" />
                  Add Key-Value Pair
                </Button>
              </div>
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>
    </div>
  );
}
