"use client";
import { useTranslation } from "react-i18next";
import { useState } from "react";
import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Source, Summary, SyncResult, useExternalMutation, useExternalQuery } from "./useExternalAgents";
import { SetupCard, ModeControl } from "./ConnectionDialogs";
import { PoliciesPanel, TeamOption, selectClass } from "./PoliciesPanel";
import { BotsPanel } from "./BotsPanel";
import { DecisionsPanel } from "./DecisionsPanel";
import { displayTime, memberMappings } from "./forms";

const tabs = ["bots", "policies", "decisions"] as const;

export default function ExternalAgentsPanel({ token, teams }: { token: string; teams: TeamOption[] }) {
  const { t } = useTranslation();
  const sources = useExternalQuery<Source[]>(token, "/sources");
  const [sourceId, setSourceId] = useQueryState("external_source", parseAsString.withDefault(""));
  const source = sources.data?.find((item) => item.source_id === sourceId) ?? sources.data?.[0];
  const mutation = useExternalMutation(token);
  const error = sources.error || mutation.error;
  return (
    <section className="w-full space-y-6 p-6 lg:p-8 overflow-y-auto" aria-label={t("agents.external.externalAgents")}>
      <h1 className="text-2xl font-bold tracking-tight">{t("agents.external.externalAgents")}</h1>
      {sources.isPending && <p role="status">{t("agents.external.loadingConnections")}</p>}
      {error && (
        <p role="alert" className="text-destructive">
          {error.message}
        </p>
      )}
      {!sources.isPending && !source && (
        <div className="rounded-xl border border-dashed py-20 text-center space-y-4">
          <p className="text-muted-foreground">{t("agents.external.governEveryBotWithoutManualRegistration")}</p>
          <Button
            disabled={mutation.isPending}
            onClick={async () => {
              try {
                const created = await mutation.mutateAsync({ method: "POST", path: "/sources", body: {} });
                void setSourceId((created as Source).source_id);
              } catch {
                return;
              }
            }}
          >
            {t("agents.external.addGrokBot")}
          </Button>
        </div>
      )}
      {source && (
        <SourceWorkspace
          key={source.source_id}
          source={source}
          sources={sources.data ?? []}
          token={token}
          teams={teams}
          onSourceChange={(id) => {
            void setSourceId(id);
          }}
        />
      )}
    </section>
  );
}

function SourceWorkspace({
  source,
  sources,
  token,
  teams,
  onSourceChange,
}: {
  source: Source;
  sources: Source[];
  token: string;
  teams: TeamOption[];
  onSourceChange: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [tab, setTab] = useQueryState("external_tab", parseAsStringLiteral(tabs).withDefault("bots"));
  const summary = useExternalQuery<Summary>(token, "/summary", { source_id: source.source_id });
  const [setup, setSetup] = useState(false);
  const [mapping, setMapping] = useState(false);
  const lastEvent = summary.data?.last_event_at || source.last_event_at;
  const connected = Boolean(lastEvent);
  const status = connected
    ? t("agents.external.connectedBots", { count: summary.data?.discovered ?? 0 })
    : t("agents.external.waitingForFirstEvent");
  return (
    <>
      {summary.error && <p role="alert">{summary.error.message}</p>}
      {sources.length > 1 && (
        <label className="flex gap-3 items-center text-sm">
          {t("agents.external.connection")}
          <select
            className={`${selectClass} max-w-96`}
            value={source.source_id}
            onChange={(e) => {
              onSourceChange(e.target.value);
              setSetup(false);
            }}
          >
            {sources.map((item) => (
              <option key={item.source_id} value={item.source_id}>
                {item.display_name}
              </option>
            ))}
          </select>
        </label>
      )}
      <ConnectedHeader
        token={token}
        source={source}
        status={status}
        connected={connected}
        lastEvent={lastEvent}
        onSetup={() => setSetup(!setup)}
      />
      {(setup || !connected) && <SetupCard token={token} sourceId={source.source_id} status={status} />}
      <SummaryStats counts={summary.data} />
      <Tabs
        value={tab}
        onValueChange={(value) => {
          if (tabs.includes(value as typeof tab)) void setTab(value as typeof tab);
        }}
      >
        <TabsList variant="line">
          <TabsTrigger value="bots">{t("agents.external.bots")}</TabsTrigger>
          <TabsTrigger value="policies">{t("agents.external.policies")}</TabsTrigger>
          <TabsTrigger value="decisions">{t("agents.external.decisions")}</TabsTrigger>
        </TabsList>
      </Tabs>
      {tab === "bots" && <BotsPanel token={token} sourceId={source.source_id} teams={teams} />}
      {tab === "policies" && (
        <div className="space-y-4">
          <Button variant="outline" onClick={() => setMapping(true)}>
            {t("agents.external.mapMembers")}
          </Button>
          <PoliciesPanel token={token} sourceId={source.source_id} teams={teams} />
        </div>
      )}
      {tab === "decisions" && <DecisionsPanel token={token} sourceId={source.source_id} />}
      {mapping && <MemberMappingDialog token={token} sourceId={source.source_id} onClose={() => setMapping(false)} />}
    </>
  );
}

function SummaryStats({ counts }: { counts?: Summary }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap gap-8 border-y py-4 text-sm">
      <p>
        {t("agents.external.bots")} <strong className="ml-2 tabular-nums">{counts?.discovered ?? 0}</strong>
      </p>
      <p>
        {t("agents.external.needAttention")}{" "}
        <strong className="ml-2 tabular-nums">{counts?.needs_attention ?? 0}</strong>
      </p>
      <p>
        {t("agents.external.denied24h")} <strong className="ml-2 tabular-nums">{counts?.denied_24h ?? 0}</strong>
      </p>
    </div>
  );
}

function ConnectedHeader({
  token,
  source,
  status,
  connected,
  lastEvent,
  onSetup,
}: {
  token: string;
  source: Source;
  status: string;
  connected: boolean;
  lastEvent?: string | null;
  onSetup: () => void;
}) {
  const { t } = useTranslation();
  const mutation = useExternalMutation(token);
  const [sync, setSync] = useState<SyncResult | null>(null);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">{source.display_name}</h2>
          <p className="text-sm flex items-center gap-2">
            <span aria-hidden className={`size-2 rounded-full ${connected ? "bg-green-500" : "bg-amber-500"}`} />
            {status}
          </p>
          <p className="text-xs text-muted-foreground">
            {t("agents.external.lastEvent")} {displayTime(lastEvent)}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ModeControl
            value={source.mode}
            disabled={mutation.isPending}
            onChange={(mode) => {
              mutation.mutate({ method: "PATCH", path: `/sources/${source.source_id}`, body: { mode } });
            }}
          />
          <Button variant="outline" onClick={onSetup}>
            {t("agents.external.showSetup")}
          </Button>
          <Button
            variant="outline"
            disabled={mutation.isPending}
            onClick={async () => {
              try {
                setSync(
                  (await mutation.mutateAsync({
                    method: "POST",
                    path: `/sources/${source.source_id}/sync_members`,
                    body: {},
                  })) as SyncResult,
                );
              } catch {
                return;
              }
            }}
          >
            {t("agents.external.syncMembers")}
          </Button>
        </div>
      </div>
      {sync && (
        <p role="status">
          {t("agents.external.membersMatched", {
            matched: sync.matched + sync.already_mapped,
            total: sync.matched + sync.already_mapped + sync.unmatched,
          })}
        </p>
      )}
      {mutation.error && (
        <p role="alert" className="text-destructive">
          {mutation.error.message}
        </p>
      )}
    </div>
  );
}

function MemberMappingDialog({ token, sourceId, onClose }: { token: string; sourceId: string; onClose: () => void }) {
  const { t } = useTranslation();
  const [csv, setCsv] = useState("");
  const [error, setError] = useState("");
  const mutation = useExternalMutation(token);
  const submitMappings = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    const parsed = memberMappings(csv);
    if (!parsed.ok) {
      setError(parsed.error);
      return;
    }
    try {
      const body = { source_id: sourceId, mappings: parsed.mappings };
      await mutation.mutateAsync({ method: "POST", path: "/actors/bulk", body });
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
        <form className="space-y-5" onSubmit={submitMappings}>
          <label className="grid gap-2">
            {t("agents.external.memberMappings")}
            <Textarea required rows={8} value={csv} onChange={(e) => setCsv(e.target.value)} />
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
