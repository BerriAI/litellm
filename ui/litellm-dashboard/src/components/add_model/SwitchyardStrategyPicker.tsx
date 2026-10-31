import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "@/lib/toast";
import { canCreateAutoRouterForTeam, canModifyModel } from "@/utils/modelPermissions";
import TeamDropdown from "../common_components/team_dropdown";
import { modelCreateCall } from "../networking";
import { fetchAutoRouterModels, fetchAvailableModels } from "../llm_calls/fetch_models";
import SwitchyardConfig from "./SwitchyardConfig";
import { newSwitchyardConfig, switchyardConfigError, switchyardConfigSchema } from "./switchyard_config";
import type { AddAutoRouterTabProps } from "./add_auto_router_tab";

function AddSwitchyardRouter({
  accessToken,
  handleOk,
  createScope,
  userId,
  userRole,
  teams = null,
}: AddAutoRouterTabProps) {
  const [name, setName] = useState("switchyard-router");
  const [teamId, setTeamId] = useState<string | null>(null);
  const [config, setConfig] = useState(newSwitchyardConfig);
  const [saving, setSaving] = useState(false);
  const requiresTeam = createScope === "team-required";
  const actor = { userRole, userID: userId ?? null, isViewOnly: false };
  const { data: models = [] } = useQuery({
    queryKey: ["availableModels", "switchyard", accessToken, teamId],
    queryFn: () => (requiresTeam ? fetchAutoRouterModels(accessToken, teamId) : fetchAvailableModels(accessToken)),
    enabled: Boolean(accessToken) && (!requiresTeam || Boolean(teamId)),
  });
  const teamError = requiresTeam && !teamId ? "Select a team to continue" : null;
  const nameError = name.trim() ? null : "Auto router name is required";
  const memberManaged = requiresTeam && Boolean(teamId) && !canModifyModel(actor, teams, { teamId, isDbModel: true });
  const permissionError = memberManaged ? "Switchyard routers must be created by a team or proxy administrator" : null;
  const error = teamError ?? permissionError ?? nameError ?? switchyardConfigError(config);
  const save = async () => {
    if (error || saving) return;
    setSaving(true);
    try {
      const payload = {
        model_name: name,
        litellm_params: {
          model: "auto_router/switchyard_router",
          switchyard_router_config: switchyardConfigSchema.parse(config),
        },
        model_info: requiresTeam ? { team_id: teamId } : {},
      };
      await modelCreateCall(accessToken, payload);
      toast.success(`Successfully created Auto Router: ${name}`);
      handleOk();
    } catch (cause) {
      toast.fromError(cause);
    } finally {
      setSaving(false);
    }
  };
  return (
    <Card>
      <CardContent>
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <div className="space-y-2">
            <Label htmlFor="switchyard-create-name">Auto Router Name</Label>
            <Input id="switchyard-create-name" value={name} onChange={(event) => setName(event.target.value)} />
          </div>
          {requiresTeam && (
            <div className="space-y-2">
              <Label htmlFor="switchyard-team">Select Team</Label>
              <TeamDropdown
                id="switchyard-team"
                value={teamId}
                onChange={setTeamId}
                filterTeam={(team) => canCreateAutoRouterForTeam(actor, team)}
              />
            </div>
          )}
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
          <Button type="submit" disabled={Boolean(error) || saving}>
            Add Auto Router
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

export default function SwitchyardStrategyPicker({
  children,
  ...props
}: AddAutoRouterTabProps & { children: React.ReactNode }) {
  const [strategy, setStrategy] = useState("complexity");
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="auto-router-strategy">Routing strategy</Label>
        <Select value={strategy} onValueChange={(next) => next && setStrategy(next)}>
          <SelectTrigger id="auto-router-strategy" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="complexity">Complexity</SelectItem>
            <SelectItem value="switchyard">Switchyard</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {strategy === "switchyard" ? <AddSwitchyardRouter {...props} /> : children}
    </div>
  );
}
