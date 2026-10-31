import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "@/lib/toast";
import { modelPatchUpdateCall } from "../networking";
import { fetchAutoRouterModels, fetchAvailableModels } from "../llm_calls/fetch_models";
import SwitchyardConfig from "../add_model/SwitchyardConfig";
import { hydrateSwitchyardConfig, switchyardConfigError, switchyardConfigSchema } from "../add_model/switchyard_config";

interface SwitchyardDeployment {
  model_name?: string;
  litellm_params?: Record<string, unknown>;
  model_info?: { id?: string; team_id?: string | null; [key: string]: unknown };
}

export default function EditSwitchyardRouter({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  isMemberManaged = false,
}: {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: (value: SwitchyardDeployment) => void;
  modelData: SwitchyardDeployment;
  accessToken: string;
  isMemberManaged?: boolean;
}) {
  const [name, setName] = useState(modelData.model_name ?? "");
  const [config, setConfig] = useState(() =>
    hydrateSwitchyardConfig(modelData.litellm_params?.switchyard_router_config),
  );
  const [saving, setSaving] = useState(false);
  const teamId = modelData.model_info?.team_id;
  const { data: models = [] } = useQuery({
    queryKey: ["availableModels", "switchyard", accessToken, teamId],
    queryFn: () => (teamId ? fetchAutoRouterModels(accessToken, teamId) : fetchAvailableModels(accessToken)),
    enabled: isVisible && Boolean(accessToken),
  });
  const nameError = !name.trim() ? "Auto router name is required" : switchyardConfigError(config);
  const error = isMemberManaged ? "Switchyard routers must be edited by a team or proxy administrator" : nameError;
  const save = async () => {
    if (error || saving || !modelData.model_info?.id) return;
    setSaving(true);
    try {
      const params = { switchyard_router_config: switchyardConfigSchema.parse(config) };
      const patch = isMemberManaged ? { litellm_params: params } : { model_name: name, litellm_params: params };
      await modelPatchUpdateCall(accessToken, patch, modelData.model_info.id);
      onSuccess({
        ...modelData,
        model_name: isMemberManaged ? modelData.model_name : name,
        litellm_params: { ...modelData.litellm_params, ...params },
      });
      toast.success("Auto router configuration updated successfully");
      onCancel();
    } catch (cause) {
      toast.fromError(cause);
    } finally {
      setSaving(false);
    }
  };
  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[760px]">
        <DialogHeader>
          <DialogTitle>Edit Switchyard auto router</DialogTitle>
          <DialogDescription>Configure the routing algorithm and model groups</DialogDescription>
        </DialogHeader>
        <Label htmlFor="switchyard-router-name">Auto Router Name</Label>
        <Input
          id="switchyard-router-name"
          value={name}
          disabled={isMemberManaged}
          onChange={(event) => setName(event.target.value)}
        />
        <SwitchyardConfig
          value={config}
          onChange={setConfig}
          modelNames={models.filter((model) => model.mode !== "embedding").map((model) => model.model_group)}
        />
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button disabled={Boolean(error) || saving} onClick={() => void save()}>
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
