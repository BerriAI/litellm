"use client";

import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { useInvalidateLenses } from "../data/mutations";
import { lensQueries } from "../data/queries";
import { useLensApi } from "../data/LensServices";
import { findFinding, findingKey, type OwnedFinding } from "../model/inbox";
import type { Finding, Lens, Settings } from "../model/types";
import { useDialogRoute, useIssueRoute, useLensRoute } from "../route";
import { InvestigationSetup } from "../setup/InvestigationSetup";
import { MonitoringDialog } from "../setup/MonitoringDialog";
import { useLensReadiness } from "../hooks/useLensReadiness";
import { OnboardingSetup } from "../onboarding/OnboardingSetup";

import { InvestigationDetail } from "./detail/InvestigationDetail";
import { RunNowDialog } from "./detail/RunNowDialog";
import { FindingPanelBody } from "./FindingDetails";
import { InvestigationList, type InvestigationRow } from "./InvestigationList";
import { investigationScreen, type Screen, type SetupScreen } from "./investigationScreen";
import {
  InvestigationError,
  InvestigationMissing,
  InvestigationsLoadFailed,
  InvestigationsLoading,
} from "./InvestigationStates";
import { ReadinessBanner } from "./ReadinessBanner";
import { useInvestigationActions } from "./useInvestigationActions";
import { WatchAllBanner } from "./WatchAllBanner";

export interface InvestigationsViewProps {
  readonly readOnly?: boolean;
}

const assertNever = (value: never): never => {
  throw new Error(`Unhandled screen ${JSON.stringify(value)}`);
};

interface BannersProps {
  readonly error: Error | null | undefined;
  readonly activityReady: boolean | null;
  readonly flush: boolean;
  readonly refresh: () => void;
}

function Banners({ error, activityReady, flush, refresh }: BannersProps) {
  if (!error && activityReady === null) return null;
  return (
    <div className={cn("flex flex-col gap-3", flush && "p-2 pb-0")}>
      {error && <InvestigationError message={error.message} refresh={refresh} />}
      {activityReady !== null && <ReadinessBanner activityReady={activityReady} className="py-2 text-xs" />}
    </div>
  );
}

export function InvestigationsView({ readOnly = false }: InvestigationsViewProps) {
  const api = useLensApi();
  const invalidateLenses = useInvalidateLenses();
  const actions = useInvestigationActions();
  const { dialog, target, openDialog, closeDialog } = useDialogRoute();
  const { issueKey, setIssueKey } = useIssueRoute();
  const { lensId, setLensId, setTab } = useLensRoute();
  const list = useQuery(lensQueries.list(api));
  const status = useLensReadiness(true);
  const { connected } = status;
  const screenInput = { list, lensId, dialog, target };
  const screen = investigationScreen(screenInput);
  const lenses = list.data?.lenses ?? [];
  const queue = {
    api,
    lenses,
    workers: list.data?.workers ?? [],
    onConnect: () => setTab("settings"),
    onOpenLens: setLensId,
  };
  const lens = lenses.find((candidate) => candidate.id === lensId);
  const peeked = issueKey ? findFinding(lenses, issueKey) ?? null : null;
  const selectedRow = (open: Lens | undefined): InvestigationRow | null => {
    if (peeked) return { kind: "finding", ...peeked };
    return open ? { kind: "investigation", lens: open } : null;
  };
  const selectRow = (row: InvestigationRow | null) => {
    if (row?.kind === "finding") setIssueKey(findingKey(row.lens, row.finding));
    else if (row) setLensId(row.lens.id);
    else if (peeked) setIssueKey(null);
    else setLensId(null);
  };
  const reviewPeeked = async (owned: OwnedFinding, status: Finding["status"], reason: string) => {
    const saved = await actions.review(owned.lens, owned.finding, status, reason);
    if (saved) setIssueKey(null);
  };
  const dialogLens = target ? lenses.find((candidate) => candidate.id === target) : lens;
  const refresh = () => {
    actions.reset();
    void invalidateLenses();
  };
  const saveSetup = async (setup: SetupScreen, settings: Settings) => {
    if (setup.mode !== "edit" && !status.ready)
      throw new Error("Wait for recorded activity and a connected worker before starting an investigation");
    const saved = await actions.save({ id: setup.mode === "edit" ? setup.lens.id : undefined, settings });
    closeDialog();
    if (!target) setLensId(saved.id);
  };
  const bannerError = screen.kind === "failed" ? undefined : actions.error ?? list.error;
  const browsing = screen.kind === "list";
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
        if (status.loading) return <InvestigationsLoading />;
        return (
          <OnboardingSetup
            state={status}
            includeTracing={!status.hasRecordedActivity}
            className="mx-auto w-full max-w-3xl p-4 sm:p-6"
          />
        );
      case "list":
        return (
          <InvestigationList
            lenses={current.lenses}
            connected={connected}
            readOnly={readOnly}
            selected={selectedRow(current.lens)}
            onSelect={selectRow}
            onEdit={(id) => openDialog("edit", id)}
            onRunNow={(id) => openDialog("run_now", id)}
            actions={
              !readOnly && (
                <>
                  <WatchAllBanner lenses={current.lenses} />
                  <Button size="sm" className="h-8" disabled={!status.ready} onClick={() => openDialog("new")}>
                    <Plus className="size-4" /> New investigation
                  </Button>
                </>
              )
            }
          >
            {(row) =>
              row.kind === "finding" ? (
                <FindingPanelBody
                  owned={row}
                  readOnly={readOnly}
                  busy={actions.busy}
                  onReview={(owned, reviewStatus, reason) => void reviewPeeked(owned, reviewStatus, reason)}
                />
              ) : (
                <div className="min-h-0 flex-1 overflow-y-auto p-4">
                  <InvestigationDetail
                    lens={row.lens}
                    readOnly={readOnly}
                    ready={status.ready}
                    busy={actions.busy}
                    connected={connected}
                    queue={queue}
                    onEdit={() => openDialog("edit")}
                    onDuplicate={() => openDialog("duplicate")}
                    onPause={() => void actions.pause(row.lens)}
                    onEnableMonitoring={() => openDialog("monitoring")}
                    onCancelRun={() => void actions.cancelRun(row.lens)}
                    onRunNow={() => openDialog("run_now")}
                    onConnectWorker={() => setTab("settings")}
                    onReviewFinding={(owned, reviewStatus, reason) =>
                      void actions.review(owned.lens, owned.finding, reviewStatus, reason)
                    }
                  />
                </div>
              )
            }
          </InvestigationList>
        );
      case "setup":
        return (
          <InvestigationSetup
            ready={status.ready}
            mode={current.mode}
            initial={current.mode === "new" ? undefined : current.initial}
            defaultSource={!status.tracesReady && status.requestsReady ? "requests" : "traces"}
            onClose={closeDialog}
            onSave={(settings) => saveSetup(current, settings)}
          />
        );
      default:
        return assertNever(current);
    }
  };

  return (
    <section
      aria-label="Investigations"
      className={cn("flex min-h-0 w-full min-w-0 flex-1 flex-col", !browsing && "gap-3 p-4")}
    >
      <Banners
        error={bannerError}
        activityReady={showReadiness ? status.activityReady : null}
        flush={browsing}
        refresh={refresh}
      />
      {content(screen)}
      {dialog === "run_now" && dialogLens && (
        <RunNowDialog
          lens={dialogLens}
          busy={actions.busy}
          onClose={closeDialog}
          onRun={async (request) => {
            const started = await actions.startRun(dialogLens, request);
            if (started) {
              closeDialog();
              setLensId(dialogLens.id);
            }
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
