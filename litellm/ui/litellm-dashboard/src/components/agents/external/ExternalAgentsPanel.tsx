"use client";
import { useTranslation } from "react-i18next";

import { useState } from "react";
import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import { Bot, ShieldCheck, Users, UserRoundSearch, ShieldAlert, PlugZap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Source, Summary, useExternalMutation, useExternalQuery } from "./useExternalAgents";
import { ConnectionDialog, HookUrlDialog, ModeControl } from "./ConnectionDialogs";
import { PoliciesPanel, TeamOption, selectClass } from "./PoliciesPanel";
import { BotsPanel } from "./BotsPanel";
import { DecisionsPanel } from "./DecisionsPanel";
import { displayTime, memberMappings } from "./forms";

const tabs = ["bots", "policies", "decisions"] as const;
const summaryCards = [
  { key: "discovered", label: "agents.external.botsDiscovered", icon: Bot },
  { key: "covered_by_team_policy", label: "agents.external.coveredByTeamPolicy", icon: ShieldCheck },
  { key: "needs_owner", label: "agents.external.needAnOwner", icon: Users },
  { key: "unmapped_actors", label: "agents.external.unmappedMembers", icon: UserRoundSearch },
  { key: "denied_24h", label: "agents.external.denied24h", icon: ShieldAlert },
] as const;

export default function ExternalAgentsPanel({ token, teams }: { token: string; teams: TeamOption[] }) {
  const { t } = useTranslation();

  const sources = useExternalQuery<Source[]>(token, "/sources");
  const [sourceId, setSourceId] = useQueryState("external_source", parseAsString.withDefault(""));
  const [tab, setTab] = useQueryState("external_tab", parseAsStringLiteral(tabs).withDefault("bots"));
  const source = sources.data?.find((item) => item.source_id === sourceId) ?? sources.data?.[0];
  const summary = useExternalQuery<Summary>(token, "/summary", { source_id: source?.source_id }, Boolean(source));
  const [connecting, setConnecting] = useState(false);
  const [rotating, setRotating] = useState(false);
  const [hookSource, setHookSource] = useState<Source | null>(null);
  const [mapping, setMapping] = useState(false);

  const connected = (created: Source) => {
    setConnecting(false);
    setRotating(false);
    void setSourceId(created.source_id);
    setHookSource(created);
  };
  return (
    <section className="w-full space-y-6 p-6 lg:p-8 overflow-y-auto" aria-label={t("agents.external.externalAgents")}>
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{t("agents.external.externalAgents")}</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t("agents.external.connectGrokBotOnceDiscoverEveryBotInheritTeamPoliciesAndResolveOnlyExceptions")}
          </p>
        </div>
        <Button onClick={() => setConnecting(true)}>
          <PlugZap className="size-4" />
          {t("agents.external.connectGrokBot")}
        </Button>
      </header>
      {sources.isPending && <p role="status">{t("agents.external.loadingConnections")}</p>}
      {(sources.error || summary.error) && (
        <div role="alert" className="rounded-lg border border-destructive/30 p-4 text-destructive">
          <p>{(sources.error || summary.error)?.message}</p>
          <Button
            variant="outline"
            onClick={() => {
              void sources.refetch();
              if (source) void summary.refetch();
            }}
          >
            {t("agents.external.retry")}
          </Button>
        </div>
      )}
      {!sources.isPending && !sources.error && !source && (
        <div className="rounded-xl border border-dashed bg-card px-8 py-20 text-center space-y-4">
          <PlugZap className="mx-auto size-10 text-muted-foreground" />
          <h2 className="text-xl font-semibold">{t("agents.external.governEveryBotWithoutManualRegistration")}</h2>
          <p className="mx-auto max-w-xl text-sm text-muted-foreground">
            {t("agents.external.grokBotSendsASignedCheckBeforeModelAndToolCallsLiteLLMAppliesYourTeamsPolicyAnd")}
          </p>
          <p className="text-sm text-muted-foreground">
            {t("agents.external.startInShadowModeToSeeExceptionsBeforeEnforcingYourPolicies")}
          </p>
        </div>
      )}
      {source && (
        <>
          <SourceOverview
            token={token}
            source={source}
            connections={sources.data ?? []}
            lastEvent={summary.data?.last_event_at}
            counts={summary.data}
            onSourceChange={(id) => {
              void setSourceId(id);
            }}
            onMap={() => setMapping(true)}
            onRotate={() => setRotating(true)}
            onShowUrl={() => setHookSource(source)}
          />
          <Tabs
            value={tab}
            onValueChange={(value) => {
              if (tabs.includes(value as typeof tab)) void setTab(value as typeof tab);
            }}
          >
            <TabsList variant="line">
              <TabsTrigger value="bots">{t("agents.external.botsExceptions")}</TabsTrigger>
              <TabsTrigger value="policies">{t("agents.external.policies")}</TabsTrigger>
              <TabsTrigger value="decisions">{t("agents.external.decisions")}</TabsTrigger>
            </TabsList>
          </Tabs>
          {tab === "bots" && (
            <BotsPanel key={source.source_id} token={token} sourceId={source.source_id} teams={teams} />
          )}
          {tab === "policies" && (
            <PoliciesPanel key={source.source_id} token={token} sourceId={source.source_id} teams={teams} />
          )}
          {tab === "decisions" && <DecisionsPanel key={source.source_id} token={token} sourceId={source.source_id} />}
        </>
      )}
      {(connecting || (rotating && source)) && (
        <ConnectionDialog
          token={token}
          source={rotating ? source : undefined}
          onClose={() => {
            setConnecting(false);
            setRotating(false);
          }}
          onConnected={connected}
        />
      )}
      {hookSource && <HookUrlDialog source={hookSource} onClose={() => setHookSource(null)} />}
      {mapping && source && (
        <MemberMappingDialog token={token} sourceId={source.source_id} onClose={() => setMapping(false)} />
      )}
    </section>
  );
}

function MemberMappingDialog({ token, sourceId, onClose }: { token: string; sourceId: string; onClose: () => void }) {
  const { t } = useTranslation();

  const [csv, setCsv] = useState("");
  const [error, setError] = useState("");
  const mutation = useExternalMutation(token);
  const submitForm = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    const parsed = memberMappings(csv);
    if (!parsed.ok) {
      setError(parsed.error);
      return;
    }
    try {
      await mutation.mutateAsync({
        method: "POST",
        path: "/actors/bulk",
        body: { source_id: sourceId, mappings: parsed.mappings },
      });
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not map members.");
    }
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent showCloseButton={false} className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("agents.external.mapMembers")}</DialogTitle>
          <DialogDescription>
            {t("agents.external.mapGrokBotMembersToExistingLiteLLMUsersPasteUpTo1000LinesWithoutAHeader")}
          </DialogDescription>
        </DialogHeader>
        <form className="space-y-5" onSubmit={submitForm}>
          <label className="grid gap-2">
            {t("agents.external.memberMappings")}
            <Textarea
              required
              rows={8}
              value={csv}
              placeholder={t("agents.external.grokbotmemberidlitellmuserid")}
              onChange={(e) => setCsv(e.target.value)}
            />
          </label>
          {error && (
            <p role="alert" className="text-destructive">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={onClose}>
              {t("agents.external.cancel")}
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {t("agents.external.saveMappings")}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function SourceOverview({
  token,
  source,
  connections,
  lastEvent,
  counts,
  onSourceChange,
  onMap,
  onRotate,
  onShowUrl,
}: {
  token: string;
  source: Source;
  connections: Source[];
  lastEvent?: string | null;
  counts?: Summary;
  onSourceChange: (id: string) => void;
  onMap: () => void;
  onRotate: () => void;
  onShowUrl: () => void;
}) {
  const { t } = useTranslation();
  const mutation = useExternalMutation(token);
  return (
    <>
      <div className="rounded-xl border bg-card p-5 space-y-4">
        {connections && connections.length > 1 && (
          <label className="flex items-center gap-3 text-sm">
            {t("agents.external.connection")}
            <select
              className={`${selectClass} max-w-96`}
              value={source.source_id}
              onChange={(e) => {
                onSourceChange(e.target.value);
              }}
            >
              {connections.map((item) => (
                <option key={item.source_id} value={item.source_id}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="flex flex-wrap justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">{source.display_name}</h2>
            <p className="text-sm text-muted-foreground mt-1">
              {t("agents.external.tenant")} {source.tenant_id || t("agents.external.waitingForFirstEvent")}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {t("agents.external.lastEvent")} {displayTime(lastEvent)}
            </p>
          </div>
          <div className="flex flex-wrap items-start gap-2">
            <Button variant="outline" onClick={() => onMap()}>
              {t("agents.external.mapMembers")}
            </Button>
            <Button variant="outline" onClick={() => onRotate()}>
              {t("agents.external.rotateSecret")}
            </Button>
            <Button variant="outline" onClick={() => onShowUrl()}>
              {t("agents.external.showHookURL")}
            </Button>
          </div>
        </div>
        <div className="border-t pt-4">
          <p className="text-sm font-medium mb-2">{t("agents.external.governanceMode")}</p>
          <ModeControl
            value={source.mode}
            disabled={mutation.isPending}
            onChange={(mode) => {
              mutation.mutate({ method: "PATCH", path: `/sources/${source.source_id}`, body: { mode } });
            }}
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-5">
        {summaryCards.map(({ key, label, icon: Icon }) => (
          <div key={key} className="rounded-xl border bg-card p-4">
            <div className="flex items-center justify-between gap-2 text-muted-foreground">
              <p className="text-xs font-medium">{t(label)}</p>
              <Icon className="size-4 shrink-0" />
            </div>
            <p className="mt-3 text-3xl font-semibold tabular-nums">{counts ? counts[key].toLocaleString() : "…"}</p>
          </div>
        ))}
      </div>

      {mutation.error && (
        <p role="alert" className="text-destructive">
          {mutation.error.message}
        </p>
      )}
    </>
  );
}
