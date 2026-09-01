"use client";

import { Edit2, Plus, Trash2 } from "lucide-react";
import { type FormEvent, useMemo, useState } from "react";

import {
  AutoRouterDeployment,
  useFusionRouters,
  useInvalidateFusionRouters,
  usePlainModelGroups,
} from "@/app/(dashboard)/hooks/models/useModels";
import TeamDropdown from "@/components/common_components/team_dropdown";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { type Model, type Team, modelCreateCall, modelDeleteCall, modelPatchUpdateCall } from "@/components/networking";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "@/lib/toast";
import { canModifyModel, type ModelWriteScope } from "@/utils/modelPermissions";

import {
  FusionFormValue,
  FusionPreset,
  fusionConfigError,
  fusionModelPayload,
  parseFusionConfig,
  presetFailureMode,
} from "./fusionModelConfig";

interface FusionModelsPanelProps {
  accessToken: string;
  userRole: string;
  userID: string | null;
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
  const [aggregatorModel, setAggregatorModel] = useState(initialConfig.aggregator_model);
  const [minSuccessful, setMinSuccessful] = useState(initialConfig.min_successful_panelists);
  const [timeoutSeconds, setTimeoutSeconds] = useState(initialConfig.panel_timeout_seconds);
  const [maxCandidateChars, setMaxCandidateChars] = useState(initialConfig.max_candidate_chars);
  const [preset, setPreset] = useState<FusionPreset>(
    initialConfig.on_quorum_failure === "aggregator_only" ? "resilient" : "quality",
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const requiresTeamScope = !editing && createScope === "team-required";
  const modelOptions = availableModels.map((model) => ({ label: model, value: model }));

  const handlePanelsChanged = (models: string[]) => {
    const limited = models.slice(0, 6);
    setPanelModels(limited);
    setMinSuccessful((current) => Math.min(Math.max(1, current), Math.max(1, limited.length)));
  };

  const applyPreset = (nextPreset: FusionPreset) => {
    setPreset(nextPreset);
    setMinSuccessful(Math.min(2, Math.max(1, panelModels.length)));
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const value: FusionFormValue = {
      model_name: modelName,
      team_id: teamID,
      panel_models: panelModels,
      aggregator_model: aggregatorModel,
      min_successful_panelists: minSuccessful,
      panel_timeout_seconds: timeoutSeconds,
      max_candidate_chars: maxCandidateChars,
      on_quorum_failure: presetFailureMode(preset),
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
            Every request runs the panel in parallel, then the aggregator returns one normal model response or tool
            call.
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
            <Select value={preset} onValueChange={(value) => applyPreset(value as FusionPreset)}>
              <SelectTrigger id="fusion-preset">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="quality" label="Quality First">
                  <div>
                    <div className="font-medium">Quality First</div>
                    <div className="text-xs text-muted-foreground">
                      Fail the request when the panel quorum is missed.
                    </div>
                  </div>
                </SelectItem>
                <SelectItem value="resilient" label="High Availability">
                  <div>
                    <div className="font-medium">High Availability</div>
                    <div className="text-xs text-muted-foreground">
                      Let the aggregator answer alone when the panel quorum is missed.
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
              placeholder="Select 2–6 independent models"
              emptyText="Add regular model deployments before creating a Fusion model."
            />
            <p className="text-xs text-muted-foreground">
              Panel models see the full request and function schemas, but their tool proposals never execute.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="fusion-aggregator">Aggregator model</Label>
            <Select value={aggregatorModel} onValueChange={(value) => setAggregatorModel(value ?? "")}>
              <SelectTrigger id="fusion-aggregator">
                <SelectValue placeholder="Select the model that produces the final response" />
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
              This model synthesizes the panel instead of choosing a winner. Only its response reaches the client.
            </p>
          </div>

          <div className="rounded-md border">
            <button
              type="button"
              className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium"
              onClick={() => setAdvancedOpen((current) => !current)}
            >
              Advanced settings
              <span className="text-xs text-muted-foreground">{advancedOpen ? "Hide" : "Show"}</span>
            </button>
            {advancedOpen && (
              <div className="grid gap-4 border-t p-4 sm:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="fusion-quorum">Successful panelists</Label>
                  <Input
                    id="fusion-quorum"
                    type="number"
                    min={1}
                    max={Math.max(1, panelModels.length)}
                    value={minSuccessful}
                    onChange={(event) => setMinSuccessful(Number(event.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="fusion-timeout">Panel timeout (seconds)</Label>
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

export function FusionModelsPanel({ accessToken, userRole, userID, teams, createScope }: FusionModelsPanelProps) {
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
    canModifyModel({ userRole, userID }, teams, {
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
            Run several models independently on every model turn, then have one aggregator synthesize the final answer
            or tool call. Your agent, coding harness, and tool loop stay unchanged.
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
              <th className="px-4 py-3 font-medium">Panel</th>
              <th className="px-4 py-3 font-medium">Aggregator</th>
              <th className="px-4 py-3 font-medium">Policy</th>
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
                  <td className="px-4 py-3 text-muted-foreground">
                    {config.panel_models.join(", ")} ({config.min_successful_panelists} required)
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">{config.aggregator_model}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {config.on_quorum_failure === "fail" ? "Quality First" : "High Availability"}
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
                <td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">
                  {canCreate
                    ? "No Fusion models yet. Create one after adding at least two regular model groups."
                    : "No Fusion models are available."}
                </td>
              </tr>
            )}
            {isLoading && (
              <tr>
                <td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">
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
