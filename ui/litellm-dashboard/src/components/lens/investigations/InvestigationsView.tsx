"use client";

import type { ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

import { lensKeys, lensQueries } from "../api/queries";
import { useLensApi } from "../LensServices";
import { findFinding, findingKey, type OwnedFinding } from "../model/inbox";
import { activityCheck, readiness } from "../model/status";
import type { Finding, Settings } from "../model/types";
import { useDialogRoute, useIssueRoute, useLensRoute } from "../route";
import { InvestigationSetup } from "../setup/InvestigationSetup";
import { MonitoringDialog } from "../setup/MonitoringDialog";
import { useWorkerConnected } from "../useWorkerConnected";

import { InvestigationDetail } from "./detail/InvestigationDetail";
import { RunNowDialog } from "./detail/RunNowDialog";
import { FindingPanel } from "./FindingDetails";
import { InvestigationList } from "./InvestigationList";
import { investigationScreen, type Screen, type SetupScreen } from "./investigationScreen";
import {
  InvestigationError,
  InvestigationMissing,
  InvestigationsLoadFailed,
  InvestigationsLoading,
} from "./InvestigationStates";
import { InvestigationsWelcome } from "./InvestigationsWelcome";
import { ReadinessBanner } from "./ReadinessBanner";
import { useInvestigationActions } from "./useInvestigationActions";
import { WatchAllBanner } from "./WatchAllBanner";

export interface InvestigationsViewProps {
  readonly accessToken: string;
  readonly readOnly?: boolean;
}

const assertNever = (value: never): never => {
  throw new Error(`Unhandled screen ${JSON.stringify(value)}`);
};

export function InvestigationsView({ accessToken, readOnly = false }: InvestigationsViewProps) {
  const api = useLensApi();
  const client = useQueryClient();
  const actions = useInvestigationActions();
  const { dialog, target, openDialog, closeDialog } = useDialogRoute();
  const { issueKey, setIssueKey } = useIssueRoute();
  const { lensId, setLensId, setTab } = useLensRoute();
  const list = useQuery(lensQueries.list(api));
  const loaded = !list.isLoading && !list.error;
  const connected = useWorkerConnected(list.data?.workers);
  const activity = useQuery(lensQueries.activity(api, loaded));
  const status = readiness(activity.data, activity.error, connected, list.error);
  const screenInput = { list, lensId, dialog, target };
  const screen = investigationScreen(screenInput);
  const lenses = list.data?.lenses ?? [];
  const lens = lenses.find((candidate) => candidate.id === lensId);
  const peeked = issueKey ? findFinding(lenses, issueKey) ?? null : null;
  const selectPeeked = (owned: OwnedFinding | null) =>
    setIssueKey(owned ? findingKey(owned.lens, owned.finding) : null);
  const reviewPeeked = async (owned: OwnedFinding, status: Finding["status"], reason: string) => {
    const saved = await actions.review(owned.lens, owned.finding, status, reason);
    if (saved) setIssueKey(null);
  };
  const dialogLens = target ? lenses.find((candidate) => candidate.id === target) : lens;
  const refresh = () => {
    actions.reset();
    void client.invalidateQueries({ queryKey: lensKeys.list(api.scope) });
    void client.invalidateQueries({ queryKey: lensKeys.histories() });
  };
  const saveSetup = async (setup: SetupScreen, settings: Settings) => {
    if (setup.mode !== "edit" && !status.ready)
      throw new Error("Wait for recorded activity and a connected worker before starting an investigation");
    const saved = await actions.save({ id: setup.mode === "edit" ? setup.lens.id : undefined, settings });
    closeDialog();
    if (!target) setLensId(saved.id);
  };
  const bannerError = screen.kind === "failed" ? undefined : actions.error ?? list.error;
  const browsing = screen.kind === "list" || screen.kind === "detail";
  const showReadiness = browsing && !readOnly && !status.ready;

  const content = (current: Screen): ReactNode => {
    switch (current.kind) {
      case "loading":
        return <InvestigationsLoading />;
      case "failed":
        return <InvestigationsLoadFailed queryError={current.error} refresh={refresh} />;
      case "missing":
        return <InvestigationMissing selectLens={setLensId} />;
      case "welcome":
        return (
          <InvestigationsWelcome
            readiness={status}
            activity={activityCheck(activity)}
            readOnly={readOnly}
            onRetry={() => {
              void activity.refetch();
              refresh();
            }}
            onConnect={() => setTab("settings")}
            onCreate={() => openDialog("new")}
          />
        );
      case "list":
        return (
          <InvestigationList
            lenses={current.lenses}
            connected={connected}
            readOnly={readOnly}
            selectedFinding={peeked}
            onSelectFinding={selectPeeked}
            onOpen={setLensId}
            onEdit={(id) => openDialog("edit", id)}
            onRunNow={(id) => openDialog("run_now", id)}
            actions={
              !readOnly && (
                <>
                  <WatchAllBanner lenses={current.lenses} />
                  <Button size="sm" className="h-7" disabled={!status.ready} onClick={() => openDialog("new")}>
                    <Plus className="size-4" /> New investigation
                  </Button>
                </>
              )
            }
          >
            <FindingPanel
              readOnly={readOnly}
              busy={actions.busy}
              accessToken={accessToken}
              onReview={(owned, status, reason) => void reviewPeeked(owned, status, reason)}
            />
          </InvestigationList>
        );
      case "detail":
        return (
          <>
            <header>
              <Button variant="ghost" size="sm" className="-ml-3" onClick={() => setLensId(null)}>
                <ArrowLeft className="size-4" /> Back
              </Button>
            </header>
            <InvestigationDetail
              lens={current.lens}
              readOnly={readOnly}
              ready={status.ready}
              busy={actions.busy}
              connected={connected}
              accessToken={accessToken}
              onEdit={() => openDialog("edit")}
              onDuplicate={() => openDialog("duplicate")}
              onPause={() => void actions.pause(current.lens)}
              onEnableMonitoring={() => openDialog("monitoring")}
              onCancelRun={() => void actions.cancelRun(current.lens)}
              onRunNow={() => openDialog("run_now")}
              onReviewFinding={(owned, status, reason) =>
                void actions.review(owned.lens, owned.finding, status, reason)
              }
            />
          </>
        );
      case "setup":
        return (
          <InvestigationSetup
            ready={status.ready}
            mode={current.mode}
            initial={current.mode === "new" ? undefined : current.initial}
            defaultSource={!status.tracesReady && status.requestsReady ? "requests" : "traces"}
            accessToken={accessToken}
            onClose={closeDialog}
            onSave={(settings) => saveSetup(current, settings)}
          />
        );
      default:
        return assertNever(current);
    }
  };

  return (
    <section aria-label="Investigations" className="flex w-full min-w-0 flex-1 flex-col gap-3">
      {bannerError && <InvestigationError message={bannerError.message} refresh={refresh} />}
      {showReadiness && <ReadinessBanner activityReady={status.activityReady} className="py-2 text-xs" />}
      {content(screen)}
      {dialog === "run_now" && dialogLens && (
        <RunNowDialog
          lens={dialogLens}
          busy={actions.busy}
          onClose={closeDialog}
          onRun={async (request) => {
            await actions.startRun(dialogLens, request);
            closeDialog();
          }}
        />
      )}
      {dialog === "monitoring" && dialogLens && (
        <MonitoringDialog
          settings={dialogLens.settings}
          ready={status.ready}
          onClose={closeDialog}
          onSave={async (settings) => {
            await actions.save({ id: dialogLens.id, settings });
          }}
        />
      )}
    </section>
  );
}
