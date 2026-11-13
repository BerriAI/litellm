import { useTranslation } from "react-i18next";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Policy, PolicyInput, useExternalMutation, useExternalQuery } from "./useExternalAgents";

export type TeamOption = { team_id: string; team_alias?: string | null };
export const selectClass = "h-9 rounded-md border border-input bg-background px-3 text-sm min-w-40";

export function PoliciesPanel({ token, sourceId, teams }: { token: string; sourceId: string; teams: TeamOption[] }) {
  const { t } = useTranslation();

  const query = useExternalQuery<Policy[]>(token, "/policies", { source_id: sourceId });
  const mutation = useExternalMutation(token);
  const [editing, setEditing] = useState<PolicyInput | null>(null);
  const [tool, setTool] = useState("");
  const defaultPolicy = query.data?.find((policy) => !policy.team_id);
  const newPolicy: PolicyInput = {
    source_id: sourceId,
    team_id: "",
    blocked_tools: [],
    allow_unmapped_actors: true,
    require_owner: false,
  };
  const submitForm = async (event: React.FormEvent<HTMLFormElement>) => {
    if (!editing) return;
    event.preventDefault();
    const blocked = [
      ...new Set([
        ...(editing.blocked_tools ?? []),
        ...tool
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean),
      ]),
    ];
    try {
      await mutation.mutateAsync({
        method: "PUT",
        path: "/policies",
        body: { ...editing, blocked_tools: blocked },
      });
      setEditing(null);
    } catch {
      return;
    }
  };
  const addTool = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!editing) return;
    if (event.key !== "Enter" && event.key !== ",") return;
    event.preventDefault();
    if (tool.trim())
      setEditing({
        ...editing,
        blocked_tools: [...new Set([...(editing.blocked_tools ?? []), tool.trim()])],
      });
    setTool("");
  };
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">{t("agents.external.teamDefaultPolicies")}</h2>
          <p className="text-sm text-muted-foreground">
            {t("agents.external.newBotsInheritTheirTeamPolicyOtherwiseTheSourceDefaultApplies")}
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => {
            setTool("");
            setEditing(newPolicy);
          }}
        >
          {t("agents.external.addTeamPolicy")}
        </Button>
      </div>
      {query.isPending && <p role="status">{t("agents.external.loadingPolicies")}</p>}
      {query.error && (
        <p role="alert" className="text-destructive">
          {query.error.message}
        </p>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        <PolicyCard
          title={t("agents.external.organizationSourceDefault")}
          policy={defaultPolicy}
          onEdit={() => {
            setTool("");
            setEditing(
              defaultPolicy ?? {
                source_id: sourceId,
                team_id: null,
                blocked_tools: [],
                allow_unmapped_actors: true,
                require_owner: false,
              },
            );
          }}
        />
        {query.data
          ?.filter((policy) => policy.team_id)
          .map((policy) => (
            <PolicyCard
              key={policy.policy_id}
              title={teams.find((team) => team.team_id === policy.team_id)?.team_alias || policy.team_id!}
              policy={policy}
              onEdit={() => {
                setTool("");
                setEditing(policy);
              }}
            />
          ))}
      </div>
      {editing && (
        <Dialog
          open
          onOpenChange={(open) => {
            if (!open) setEditing(null);
          }}
        >
          <DialogContent showCloseButton={false} className="sm:max-w-xl">
            <DialogHeader>
              <DialogTitle>{t("agents.external.editPolicy")}</DialogTitle>
              <DialogDescription>
                {t("agents.external.changesApplyToAllBotsUsingThisPolicyOtherWorkersRefreshWithin20Seconds")}
              </DialogDescription>
            </DialogHeader>
            <form className="space-y-5" onSubmit={submitForm}>
              {editing.team_id !== null && (
                <label className="grid gap-2">
                  {t("agents.external.team")}
                  <select
                    required
                    aria-label={t("agents.external.policyTeam")}
                    className={selectClass}
                    value={editing.team_id ?? ""}
                    disabled={"policy_id" in editing}
                    onChange={(e) => setEditing({ ...editing, team_id: e.target.value })}
                  >
                    <option value="">{t("agents.external.selectATeam")}</option>
                    {teams.map((team) => (
                      <option key={team.team_id} value={team.team_id}>
                        {team.team_alias || team.team_id}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="space-y-2">
                <label htmlFor="blocked-tool">{t("agents.external.blockedTools")}</label>
                <div className="flex flex-wrap gap-2">
                  {editing.blocked_tools?.map((name) => (
                    <div key={name} className="flex items-center gap-1 rounded-md border px-2">
                      <span>{name}</span>
                      <Button
                        type="button"
                        size="xs"
                        variant="ghost"
                        aria-label={t("agents.external.removeValue0", { value0: name })}
                        onClick={() =>
                          setEditing({
                            ...editing,
                            blocked_tools: editing.blocked_tools?.filter((item) => item !== name),
                          })
                        }
                      >
                        {t("agents.external.remove")}
                      </Button>
                    </div>
                  ))}
                </div>
                <Input
                  id="blocked-tool"
                  value={tool}
                  placeholder={t("agents.external.typeAToolNameAndPressEnter")}
                  onChange={(e) => setTool(e.target.value)}
                  onKeyDown={addTool}
                />
                <p className="text-xs text-muted-foreground">
                  {t("agents.external.matchShellCallMcpToolOrMCPServerNamesCaseinsensitively")}
                </p>
              </div>
              <label className="flex items-center justify-between">
                {t("agents.external.allowUnmappedMembers")}
                <Switch
                  checked={editing.allow_unmapped_actors ?? true}
                  onCheckedChange={(checked) => setEditing({ ...editing, allow_unmapped_actors: checked })}
                />
              </label>
              <label className="flex items-center justify-between">
                {t("agents.external.requireOwner")}
                <Switch
                  checked={editing.require_owner ?? false}
                  onCheckedChange={(checked) => setEditing({ ...editing, require_owner: checked })}
                />
              </label>
              {mutation.error && (
                <p role="alert" className="text-destructive">
                  {mutation.error.message}
                </p>
              )}
              <div className="flex justify-end gap-2">
                <Button type="button" variant="outline" onClick={() => setEditing(null)}>
                  {t("agents.external.cancel")}
                </Button>
                <Button type="submit" disabled={mutation.isPending}>
                  {t("agents.external.savePolicy")}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}

function PolicyCard({ title, policy, onEdit }: { title: string; policy?: Policy; onEdit: () => void }) {
  const { t } = useTranslation();

  return (
    <div className="rounded-xl border bg-card p-5 space-y-4">
      <div className="flex justify-between items-start gap-2">
        <h3 className="font-semibold break-all">{title}</h3>
        <Button variant="outline" size="sm" onClick={onEdit}>
          {t("agents.external.editPolicy")}
        </Button>
      </div>
      <div>
        <p className="text-xs text-muted-foreground mb-2">{t("agents.external.blockedTools")}</p>
        <div className="flex flex-wrap gap-2">
          {policy?.blocked_tools.length ? (
            policy.blocked_tools.map((name) => (
              <Badge variant="secondary" key={name}>
                {name}
              </Badge>
            ))
          ) : (
            <span className="text-sm text-muted-foreground">{t("agents.external.noneAllToolsAllowed")}</span>
          )}
        </div>
      </div>
      <p className="flex items-center gap-2 text-sm">
        {t("agents.external.unmappedMembers")}
        <strong>
          {policy?.allow_unmapped_actors !== false ? t("agents.external.allowed") : t("agents.external.blocked")}
        </strong>
      </p>
      <p className="flex items-center gap-2 text-sm">
        {t("agents.external.owner")}
        <strong>{policy?.require_owner ? t("agents.external.required") : t("agents.external.optional")}</strong>
      </p>
    </div>
  );
}
