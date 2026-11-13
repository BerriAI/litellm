import { useTranslation } from "react-i18next";
import { useState } from "react";
import { parseAsBoolean, parseAsString, useQueryStates } from "nuqs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { Bot, BulkInput, useExternalMutation, useExternalQuery } from "./useExternalAgents";
import { TeamOption, selectClass } from "./PoliciesPanel";
import { displayTime, pageParser } from "./forms";

const filters = {
  exceptions_only: parseAsBoolean.withDefault(true),
  bot_search: parseAsString.withDefault(""),
  bot_team: parseAsString.withDefault(""),
  bot_page: pageParser,
};
const pageSize = 20;
const selectionCap = 1000;

export function Pagination({
  page,
  count,
  onChange,
}: {
  page: number;
  count: number;
  onChange: (page: number) => void;
}) {
  const { t } = useTranslation();

  return (
    <div className="flex justify-between items-center gap-3 pt-4 text-sm text-muted-foreground">
      <span>{t("agents.external.pageResults", { page, count })}</span>
      <div className="flex gap-2">
        <Button variant="outline" disabled={page <= 1} onClick={() => onChange(page - 1)}>
          {t("agents.external.previous")}
        </Button>
        <Button variant="outline" disabled={count < pageSize} onClick={() => onChange(page + 1)}>
          {t("agents.external.next")}
        </Button>
      </div>
    </div>
  );
}

export function BotsPanel({ token, sourceId, teams }: { token: string; sourceId: string; teams: TeamOption[] }) {
  const { t } = useTranslation();

  const [state, setState] = useQueryStates(filters);
  const page = state.bot_page;
  const query = useExternalQuery<Bot[]>(token, "", {
    source_id: sourceId,
    exceptions_only: state.exceptions_only,
    search: state.bot_search || undefined,
    team_id: state.bot_team || undefined,
    limit: pageSize,
    offset: (page - 1) * pageSize,
  });
  const bots = query.data ?? [];
  const mutation = useExternalMutation(token);
  const [selection, setSelection] = useState<{ scope: string; ids: string[] }>({ scope: "", ids: [] });
  const scope = JSON.stringify(state);
  const selected = selection.scope === scope ? selection.ids.filter((id) => bots.some((bot) => bot.id === id)) : [];
  const setSelected = (ids: string[]) => setSelection({ scope, ids });
  const [assign, setAssign] = useState<"team" | "owner" | null>(null);
  const [value, setValue] = useState("");
  const selectedBots = bots.filter((bot) => selected.includes(bot.id));
  const bulk = async (input: BulkInput) => {
    try {
      await mutation.mutateAsync({ method: "PATCH", path: "/bulk", body: input });
      setSelected([]);
      setAssign(null);
    } catch {
      return;
    }
  };
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 font-medium text-sm">
          <Switch
            checked={state.exceptions_only}
            onCheckedChange={(checked) => {
              setSelected([]);
              void setState({ exceptions_only: checked, bot_page: 1 });
            }}
          />
          {t("agents.external.exceptionsOnly")}
        </label>
        <Input
          className="max-w-72"
          aria-label={t("agents.external.searchBots")}
          placeholder={t("agents.external.searchBotsByNameOrID")}
          value={state.bot_search}
          onChange={(e) => {
            setSelected([]);
            void setState({ bot_search: e.target.value, bot_page: 1 });
          }}
        />
        <select
          aria-label={t("agents.external.filterByTeam")}
          className={selectClass}
          value={state.bot_team}
          onChange={(e) => {
            setSelected([]);
            void setState({ bot_team: e.target.value, bot_page: 1 });
          }}
        >
          <option value="">{t("agents.external.allTeams")}</option>
          {teams.map((team) => (
            <option key={team.team_id} value={team.team_id}>
              {team.team_alias || team.team_id}
            </option>
          ))}
        </select>
      </div>
      <p className="text-xs text-muted-foreground">
        {t("agents.external.resolveExceptionsInBulkInferredOwnersNeedConfirmationBecauseAnInvokingMemberMayNotOwnA")}
      </p>
      {selected.length > 0 && (
        <div
          role="region"
          aria-label={t("agents.external.bulkActions")}
          className="flex flex-wrap items-center gap-2 rounded-lg border border-primary/30 bg-primary/5 p-3"
        >
          <strong className="text-sm mr-2">{t("agents.external.selectedCount", { count: selected.length })}</strong>
          <Button
            variant="outline"
            disabled={mutation.isPending}
            onClick={() => {
              setValue("");
              setAssign("team");
            }}
          >
            {t("agents.external.assignTeam")}
          </Button>
          <Button
            variant="outline"
            disabled={mutation.isPending}
            onClick={() => {
              setValue("");
              setAssign("owner");
            }}
          >
            {t("agents.external.assignOwner")}
          </Button>
          <Button
            variant="outline"
            disabled={mutation.isPending || selectedBots.some((bot) => !bot.owner_user_id)}
            onClick={async () => {
              try {
                for (const owner of new Set(selectedBots.map((bot) => bot.owner_user_id))) {
                  await mutation.mutateAsync({
                    method: "PATCH",
                    path: "/bulk",
                    body: {
                      agent_ids: selectedBots.filter((bot) => bot.owner_user_id === owner).map((bot) => bot.id),
                      owner_user_id: owner,
                    },
                  });
                }
                setSelected([]);
              } catch {
                return;
              }
            }}
          >
            {t("agents.external.confirmOwner")}
          </Button>
          <Button
            variant="destructive"
            disabled={mutation.isPending}
            onClick={() => bulk({ agent_ids: selected, status: "blocked" })}
          >
            {t("agents.external.block")}
          </Button>
          <Button
            variant="outline"
            disabled={mutation.isPending}
            onClick={() => bulk({ agent_ids: selected, status: "active" })}
          >
            {t("agents.external.unblock")}
          </Button>
          <Button variant="ghost" onClick={() => setSelected([])}>
            {t("agents.external.clearSelection")}
          </Button>
          <span className="text-xs text-muted-foreground">
            {t("agents.external.selectionLimit", { limit: selectionCap.toLocaleString() })}
          </span>
        </div>
      )}
      {(query.error || mutation.error) && (
        <p role="alert" className="text-destructive">
          {(query.error || mutation.error)?.message}
        </p>
      )}
      <div className="rounded-xl border bg-card overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>
                <Checkbox
                  aria-label={t("agents.external.selectAllBotsOnThisPage")}
                  checked={bots.length > 0 && bots.every((bot) => selected.includes(bot.id))}
                  disabled={!bots.length || mutation.isPending}
                  onCheckedChange={(checked) =>
                    setSelected(checked ? bots.slice(0, selectionCap).map((bot) => bot.id) : [])
                  }
                />
              </TableHead>
              {[
                t("agents.external.bot"),
                t("agents.external.team"),
                t("agents.external.owner"),
                t("agents.external.lastMember"),
                t("agents.external.lastSeen"),
                t("agents.external.requests"),
                t("agents.external.status"),
                t("agents.external.issues"),
              ].map((name) => (
                <TableHead key={name}>{name}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {
              <>
                {query.isPending && (
                  <TableRow>
                    <TableCell colSpan={9}>{t("agents.external.loadingBots")}</TableCell>
                  </TableRow>
                )}
                {!query.isPending && (
                  <>
                    {bots.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={9} className="py-12 text-center text-muted-foreground">
                          {state.exceptions_only
                            ? t("agents.external.noExceptionsNewBotsAppearAfterTheirFirstRealEvent")
                            : t("agents.external.noBotsMatchTheseFiltersBotsAreDiscoveredAutomatically")}
                        </TableCell>
                      </TableRow>
                    )}
                    {!(bots.length === 0) &&
                      bots.map((bot) => (
                        <TableRow key={bot.id} data-state={selected.includes(bot.id) ? "selected" : undefined}>
                          <TableCell>
                            <Checkbox
                              aria-label={t("agents.external.selectValue0", {
                                value0: bot.display_name || bot.external_agent_id,
                              })}
                              checked={selected.includes(bot.id)}
                              disabled={
                                mutation.isPending || (!selected.includes(bot.id) && selected.length >= selectionCap)
                              }
                              onCheckedChange={(checked) =>
                                setSelected(checked ? [...selected, bot.id] : selected.filter((id) => id !== bot.id))
                              }
                            />
                          </TableCell>
                          <TableCell className="font-medium min-w-32">
                            {bot.display_name || bot.external_agent_id}
                          </TableCell>
                          <TableCell>
                            {teams.find((team) => team.team_id === bot.team_id)?.team_alias ||
                              bot.team_id ||
                              t("agents.external.unassigned")}
                          </TableCell>
                          <TableCell>
                            <div className="space-y-1 min-w-36">
                              <span>
                                {bot.owner_display_name || bot.owner_user_id || t("agents.external.unassigned")}
                              </span>
                              {bot.owner_inferred && (
                                <div className="flex flex-wrap gap-1 items-center">
                                  <Badge variant="secondary">{t("agents.external.inferred")}</Badge>
                                  <Button
                                    size="xs"
                                    variant="outline"
                                    disabled={!bot.owner_user_id || mutation.isPending}
                                    onClick={() => bulk({ agent_ids: [bot.id], owner_user_id: bot.owner_user_id })}
                                  >
                                    {t("agents.external.confirmOwner")}
                                  </Button>
                                </div>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>{bot.last_actor_id || t("agents.external.unknown")}</TableCell>
                          <TableCell className="min-w-36">{displayTime(bot.last_seen_at)}</TableCell>
                          <TableCell>{bot.request_count.toLocaleString()}</TableCell>
                          <TableCell>
                            <Badge variant={bot.status === "blocked" ? "destructive" : "outline"}>{bot.status}</Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex flex-wrap gap-1 min-w-48">
                              {bot.issues?.length ? (
                                bot.issues.map((issue) => (
                                  <Badge
                                    key={issue}
                                    variant={issue === "denied recently" ? "destructive" : "secondary"}
                                  >
                                    {issue}
                                  </Badge>
                                ))
                              ) : (
                                <span className="text-muted-foreground">{t("agents.external.none")}</span>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                  </>
                )}
              </>
            }
          </TableBody>
        </Table>
      </div>
      <Pagination
        page={page}
        count={bots.length}
        onChange={(next) => {
          setSelected([]);
          void setState({ bot_page: next });
        }}
      />
      {assign && (
        <Dialog
          open
          onOpenChange={(open) => {
            if (!open) setAssign(null);
          }}
        >
          <DialogContent showCloseButton={false}>
            <DialogHeader>
              <DialogTitle>
                {t(assign === "team" ? "agents.external.assignTeam" : "agents.external.assignOwner")}
              </DialogTitle>
              <DialogDescription>
                {t("agents.external.updateSelectedBots", { count: selected.length })}
              </DialogDescription>
            </DialogHeader>
            <form
              className="space-y-5"
              onSubmit={(event) => {
                event.preventDefault();
                void bulk({
                  agent_ids: selected,
                  ...(assign === "team" ? { team_id: value } : { owner_user_id: value }),
                });
              }}
            >
              {assign === "team" ? (
                <label className="grid gap-2">
                  {t("agents.external.team")}
                  <select required className={selectClass} value={value} onChange={(e) => setValue(e.target.value)}>
                    <option value="">{t("agents.external.selectATeam")}</option>
                    {teams.map((team) => (
                      <option key={team.team_id} value={team.team_id}>
                        {team.team_alias || team.team_id}
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <label className="grid gap-2">
                  {t("agents.external.liteLLMUserID")}
                  <Input required value={value} onChange={(e) => setValue(e.target.value)} />
                </label>
              )}
              {mutation.error && (
                <p role="alert" className="text-destructive">
                  {mutation.error.message}
                </p>
              )}
              <div className="flex justify-end gap-2">
                <Button type="button" variant="outline" onClick={() => setAssign(null)}>
                  {t("agents.external.cancel")}
                </Button>
                <Button type="submit" disabled={mutation.isPending}>
                  {t("agents.external.applyAssignment")}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
