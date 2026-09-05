"use client";

import { Edit2, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { type FormEvent, useCallback, useMemo, useRef, useState } from "react";

import {
  AutoRouterDeployment,
  useFusionRouters,
  useInvalidateFusionRouters,
  usePlainModelGroups,
} from "@/app/(dashboard)/hooks/models/useModels";
import TeamDropdown from "@/components/common_components/team_dropdown";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { type Model, type Team, modelCreateCall, modelDeleteCall, modelPatchUpdateCall } from "@/components/networking";
import SearchToolSelector from "@/components/search_tools/SearchToolSelector";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/lib/toast";
import { canModifyModel, type ModelWriteScope } from "@/utils/modelPermissions";

import {
  FusionFormValue,
  FusionPreset,
  fusionConfigError,
  fusionModelPayload,
  parseFusionConfig,
  presetInvocation,
} from "./fusionModelConfig";

interface FusionModelsPanelProps {
  accessToken: string;
  userRole: string;
  userID: string | null;
  isViewOnly: boolean;
  teams: Team[] | null;
  createScope: ModelWriteScope;
}

interface FusionModelDialogProps {
  accessToken: string;
  availableModels: string[];
  createScope: ModelWriteScope;
  deployment: AutoRouterDeployment | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
}

const deploymentConfig = (deployment: AutoRouterDeployment | null) =>
  parseFusionConfig(deployment?.litellm_params?.fusion_router_config);

const submitButtonLabel = (saving: boolean, editing: boolean) => {
  if (saving) return "Saving…";
  return editing ? "Save Changes" : "Create Fusion Model";
};

function FusionModelDialog({
  accessToken,
  availableModels,
  createScope,
  deployment,
  open,
  onOpenChange,
  onSaved,
}: FusionModelDialogProps) {
  const editing = deployment !== null;
  const initialConfig = deploymentConfig(deployment);
  const [modelName, setModelName] = useState(deployment?.model_name ?? "");
  const [teamID, setTeamID] = useState("");
  const [panelModels, setPanelModels] = useState(initialConfig.panel_models);
  const [outerModel, setOuterModel] = useState(initialConfig.outer_model);
  const [analystModel, setAnalystModel] = useState(initialConfig.analyst_model);
  const [timeoutSeconds, setTimeoutSeconds] = useState(initialConfig.panel_timeout_seconds);
  const [maxCandidateChars, setMaxCandidateChars] = useState(initialConfig.max_candidate_chars);
  const [maxCompletionTokens, setMaxCompletionTokens] = useState(initialConfig.max_completion_tokens);
  const [temperature, setTemperature] = useState(initialConfig.temperature);
  const [reasoningEffort, setReasoningEffort] = useState(initialConfig.reasoning_effort);
  const [searchToolName, setSearchToolName] = useState(initialConfig.search_tool_name);
  const [webAccessEnabled, setWebAccessEnabled] = useState(editing ? Boolean(initialConfig.search_tool_name) : true);
  const searchDefaultsApplied = useRef(editing);
  const [maxToolCalls, setMaxToolCalls] = useState(initialConfig.max_tool_calls);
  const [preset, setPreset] = useState<FusionPreset>(initialConfig.invocation === "required" ? "always" : "auto");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const requiresTeamScope = !editing && createScope === "team-required";
  const modelOptions = availableModels.map((model) => ({ label: model, value: model }));

  const handlePanelsChanged = (models: string[]) => {
    setPanelModels(models.slice(0, 8));
  };

  const applyPreset = (nextPreset: FusionPreset) => {
    setPreset(nextPreset);
  };

  const handleSearchOptionsLoaded = useCallback((options: string[]) => {
    if (searchDefaultsApplied.current) return;
    searchDefaultsApplied.current = true;
    setWebAccessEnabled(options.length > 0);
    setSearchToolName((current) => current || options[0] || "");
  }, []);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const value: FusionFormValue = {
      model_name: modelName,
      team_id: teamID,
      outer_model: outerModel,
      panel_models: panelModels,
      analyst_model: analystModel,
      invocation: presetInvocation(preset),
      panel_timeout_seconds: timeoutSeconds,
      max_candidate_chars: maxCandidateChars,
      max_completion_tokens: maxCompletionTokens,
      temperature,
      reasoning_effort: reasoningEffort,
      search_tool_name: searchToolName,
      max_tool_calls: maxToolCalls,
      web_access_enabled: webAccessEnabled,
    };
    const validationError = fusionConfigError(value, requiresTeamScope);
    if (validationError) {
      setError(validationError);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const payload = fusionModelPayload(value, requiresTeamScope);
      if (editing) {
        const modelID = deployment.model_info?.id;
        if (!modelID) throw new Error("This Fusion model has no editable model ID.");
        await modelPatchUpdateCall(accessToken, { litellm_params: payload.litellm_params }, modelID);
        toast.success(`Updated Fusion model: ${value.model_name}`);
      } else {
        await modelCreateCall(accessToken, payload as Model);
      }
      onSaved();
      onOpenChange(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save Fusion model.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{editing ? "Configure Fusion Model" : "Add Fusion Model"}</DialogTitle>
          <DialogDescription>
            The outer model can privately ask a panel to deliberate, then uses their analysis to return the final
            response or tool call.
          </DialogDescription>
        </DialogHeader>
        <form className="space-y-5" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <Label htmlFor="fusion-name">Model name</Label>
            <Input
              id="fusion-name"
              value={modelName}
              onChange={(event) => setModelName(event.target.value)}
              placeholder="fusion/coding"
              disabled={editing}
            />
            <p className="text-xs text-muted-foreground">Clients use this name exactly like any other model.</p>
          </div>

          {requiresTeamScope && (
            <div className="space-y-2">
              <Label>Team</Label>
              <TeamDropdown value={teamID} onChange={(value) => setTeamID(value ?? "")} />
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="fusion-preset">Behavior</Label>
            <Select
              value={preset}
              onValueChange={(value) => {
                if (value === "auto" || value === "always") applyPreset(value);
              }}
            >
              <SelectTrigger id="fusion-preset" className="h-11 w-full px-3">
                <span className="flex-1 text-left font-medium">{preset === "auto" ? "Auto" : "Always deliberate"}</span>
              </SelectTrigger>
              <SelectContent align="start" className="w-[var(--radix-select-trigger-width)] min-w-[28rem]">
                <SelectItem value="auto" label="Auto" className="items-start py-3">
                  <div className="min-w-0 whitespace-normal pr-4">
                    <div className="font-medium">Auto</div>
                    <div className="mt-1 text-sm leading-5 text-muted-foreground">
                      Let the outer model deliberate only when another perspective would help.
                    </div>
                  </div>
                </SelectItem>
                <SelectItem value="always" label="Always deliberate" className="items-start py-3">
                  <div className="min-w-0 whitespace-normal pr-4">
                    <div className="font-medium">Always deliberate</div>
                    <div className="mt-1 text-sm leading-5 text-muted-foreground">
                      Force one panel deliberation on every request. Useful for evaluation and high-stakes workloads.
                    </div>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Panel models</Label>
            <MultiSelect
              options={modelOptions}
              value={panelModels}
              onValueChange={handlePanelsChanged}
              placeholder="Select 1–8 independent models"
              emptyText="Add a regular model deployment before creating a Fusion model."
            />
            <p className="text-xs text-muted-foreground">
              Panel models receive a self-contained deliberation question. They never receive or execute client tools.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="fusion-outer">Outer model</Label>
            <Select value={outerModel} onValueChange={(value) => setOuterModel(value ?? "")}>
              <SelectTrigger id="fusion-outer" className="w-full">
                <SelectValue placeholder="Select the model that talks to the client" />
              </SelectTrigger>
              <SelectContent>
                {availableModels.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              This model decides when to deliberate and is the only model that can return answers or client tool calls.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="fusion-analyst">Analyst model</Label>
            <Select
              value={analystModel || "same-as-outer"}
              onValueChange={(value) => setAnalystModel(value === "same-as-outer" ? "" : value ?? "")}
            >
              <SelectTrigger id="fusion-analyst" className="w-full">
                <SelectValue placeholder="Use the outer model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="same-as-outer">Same as outer model</SelectItem>
                {availableModels.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              The analyst compares the panel&apos;s consensus, disagreements, gaps, and unique insights. It never writes
              the final answer.
            </p>
          </div>

          <div className="space-y-3 rounded-md border p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1">
                <Label htmlFor="fusion-web-access">Web access</Label>
                <p className="text-xs text-muted-foreground">
                  Let every panel model and the analyst independently search for current evidence.
                </p>
              </div>
              <Switch
                id="fusion-web-access"
                aria-label="Web access"
                checked={webAccessEnabled}
                onCheckedChange={setWebAccessEnabled}
              />
            </div>
            {webAccessEnabled && (
              <SearchToolSelector
                accessToken={accessToken}
                value={searchToolName ? [searchToolName] : []}
                onChange={(tools) => setSearchToolName(tools.at(-1) ?? "")}
                onOptionsLoaded={handleSearchOptionsLoaded}
                placeholder="Select a Search Tool"
              />
            )}
            <p className="text-xs text-muted-foreground">
              Search runs privately and never exposes client action tools to the panel.{" "}
              <Link href="/search-tools" className="font-medium text-primary hover:underline">
                Manage Search Tools
              </Link>
            </p>
          </div>

          <div className="rounded-md border">
            <button
              type="button"
              aria-label="Advanced settings"
              className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium"
              onClick={() => setAdvancedOpen((current) => !current)}
            >
              Advanced settings
              <span className="text-xs text-muted-foreground">{advancedOpen ? "Hide" : "Show"}</span>
            </button>
            {advancedOpen && (
              <div className="grid gap-4 border-t p-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="fusion-timeout">Panel and analyst timeout (seconds)</Label>
                  <Input
                    id="fusion-timeout"
                    type="number"
                    min={1}
                    max={600}
                    value={timeoutSeconds}
                    onChange={(event) => setTimeoutSeconds(Number(event.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="fusion-candidate-limit">Candidate characters</Label>
                  <Input
                    id="fusion-candidate-limit"
                    type="number"
                    min={1000}
                    max={50000}
                    step={1000}
                    value={maxCandidateChars}
                    onChange={(event) => setMaxCandidateChars(Number(event.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="fusion-output-tokens">Internal output tokens</Label>
                  <Input
                    id="fusion-output-tokens"
                    type="number"
                    min={1}
                    max={128000}
                    value={maxCompletionTokens}
                    onChange={(event) => setMaxCompletionTokens(Number(event.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="fusion-temperature">Panel temperature</Label>
                  <Input
                    id="fusion-temperature"
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperature}
                    onChange={(event) => setTemperature(Number(event.target.value))}
                  />
                </div>
                <div className="space-y-2 sm:col-span-2">
                  <Label htmlFor="fusion-reasoning">Internal reasoning effort</Label>
                  <Select
                    value={reasoningEffort}
                    onValueChange={(value) => setReasoningEffort(value as typeof reasoningEffort)}
                  >
                    <SelectTrigger id="fusion-reasoning" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(["none", "minimal", "low", "medium", "high", "xhigh"] as const).map((effort) => (
                        <SelectItem key={effort} value={effort}>
                          {effort}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Unsupported reasoning parameters are dropped for that provider.
                  </p>
                </div>
                {webAccessEnabled && (
                  <div className="space-y-2 sm:col-span-2">
                    <Label htmlFor="fusion-tool-calls">Maximum searches per internal model</Label>
                    <Input
                      id="fusion-tool-calls"
                      type="number"
                      min={1}
                      max={16}
                      value={maxToolCalls}
                      onChange={(event) => setMaxToolCalls(Number(event.target.value))}
                    />
                    <p className="text-xs text-muted-foreground">
                      Applied separately to every panel model and the analyst. Default 4; maximum 16.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {error && <p className="text-sm text-destructive">{error}</p>}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={saving}>
              Cancel
            </Button>
            <Button type="submit" disabled={saving}>
              {submitButtonLabel(saving, editing)}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function FusionModelsPanel({
  accessToken,
  userRole,
  userID,
  isViewOnly,
  teams,
  createScope,
}: FusionModelsPanelProps) {
  const { data: deployments, isLoading } = useFusionRouters();
  const availableModels = usePlainModelGroups();
  const invalidateFusionRouters = useInvalidateFusionRouters();
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<AutoRouterDeployment | null>(null);
  const [deleting, setDeleting] = useState<AutoRouterDeployment | null>(null);
  const [deletingBusy, setDeletingBusy] = useState(false);
  const canCreate = createScope !== "forbidden";
  const modelOptions = useMemo(() => Array.from(availableModels).sort(), [availableModels]);

  const canModify = (deployment: AutoRouterDeployment) =>
    canModifyModel({ userRole, userID, isViewOnly }, teams, {
      teamId: deployment.model_info?.team_id,
      isDbModel: deployment.model_info?.db_model === true,
    });

  const handleDelete = async () => {
    const modelID = deleting?.model_info?.id;
    if (!deleting || !modelID) return;
    setDeletingBusy(true);
    try {
      await modelDeleteCall(accessToken, modelID);
      toast.success(`Deleted Fusion model: ${deleting.model_name}`);
      setDeleting(null);
      await invalidateFusionRouters();
    } catch (deleteError) {
      toast.fromError(`Failed to delete Fusion model: ${deleteError}`);
    } finally {
      setDeletingBusy(false);
    }
  };

  const rows = deployments ?? [];
  return (
    <div className="w-full space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-foreground">Fusion models</h2>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Give one model a private deliberation tool backed by an independent panel and analyst. The outer model still
            behaves like a normal model, so your agent, coding harness, and tool loop stay unchanged.
          </p>
        </div>
        {canCreate && (
          <Button onClick={() => setCreating(true)} className="shrink-0">
            <Plus /> Add Fusion Model
          </Button>
        )}
      </div>

      <div className="overflow-hidden rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left text-xs text-muted-foreground">
            <tr>
              <th className="px-4 py-3 font-medium">Name</th>
              <th className="px-4 py-3 font-medium">Outer model</th>
              <th className="px-4 py-3 font-medium">Panel</th>
              <th className="px-4 py-3 font-medium">Analyst</th>
              <th className="px-4 py-3 font-medium">Deliberation</th>
              <th className="w-24 px-4 py-3 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {rows.map((deployment) => {
              const config = deploymentConfig(deployment);
              const modifiable = canModify(deployment);
              return (
                <tr key={deployment.model_info?.id ?? deployment.model_name}>
                  <td className="px-4 py-3 font-medium">{deployment.model_name}</td>
                  <td className="px-4 py-3 text-muted-foreground">{config.outer_model}</td>
                  <td className="px-4 py-3 text-muted-foreground">{config.panel_models.join(", ")}</td>
                  <td className="px-4 py-3 text-muted-foreground">{config.analyst_model || "Same as outer"}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {config.invocation === "required" ? "Always" : "Auto"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Configure ${deployment.model_name}`}
                        disabled={!modifiable}
                        onClick={() => setEditing(deployment)}
                      >
                        <Edit2 />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Delete ${deployment.model_name}`}
                        disabled={!modifiable}
                        onClick={() => setDeleting(deployment)}
                      >
                        <Trash2 />
                      </Button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {!isLoading && rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">
                  {canCreate
                    ? "No Fusion models yet. Create one after adding at least one regular model group."
                    : "No Fusion models are available."}
                </td>
              </tr>
            )}
            {isLoading && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">
                  Loading Fusion models…
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {creating && (
        <FusionModelDialog
          key="create"
          open
          onOpenChange={setCreating}
          accessToken={accessToken}
          availableModels={modelOptions}
          createScope={createScope}
          deployment={null}
          onSaved={() => void invalidateFusionRouters()}
        />
      )}
      {editing && (
        <FusionModelDialog
          key={editing.model_info?.id ?? editing.model_name}
          open
          onOpenChange={(open) => !open && setEditing(null)}
          accessToken={accessToken}
          availableModels={modelOptions}
          createScope={createScope}
          deployment={editing}
          onSaved={() => {
            setEditing(null);
            void invalidateFusionRouters();
          }}
        />
      )}
      {deleting && (
        <DeleteResourceModal
          isOpen
          title="Delete Fusion Model"
          message={`Are you sure you want to delete "${deleting.model_name}"? Clients using this model name will start failing.`}
          resourceInformationTitle="Fusion model"
          resourceInformation={[
            { label: "Name", value: deleting.model_name ?? "" },
            { label: "ID", value: deleting.model_info?.id ?? "" },
          ]}
          onCancel={() => setDeleting(null)}
          onOk={handleDelete}
          confirmLoading={deletingBusy}
        />
      )}
    </div>
  );
}
