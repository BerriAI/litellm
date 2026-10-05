"use client";

import { Tabs as TabsPrimitive } from "@base-ui/react/tabs";
import { Activity, ScanSearch, Settings } from "lucide-react";
import { StatusDot } from "@/components/shared/StatusDot";
import { cn } from "@/lib/cva.config";
import type { InvestigationActivity } from "./model/status";
import { useWorkerConnected } from "./hooks/useWorkerConnected";
import type { LensList } from "./model/types";
import { LENS_TABS, type LensTab } from "./route";
import { frameCorner, frameTab } from "./ui/frame";

const MODE_ICONS = { traces: Activity, investigations: ScanSearch, settings: Settings } as const;

const ACTIVITY_DOT: Record<Exclude<InvestigationActivity, "idle">, { className: string; label: string }> = {
  running: { className: "bg-info motion-safe:animate-pulse", label: "An investigation is running" },
  queued: { className: "bg-muted-foreground/60", label: "An investigation is queued" },
};

function ActivityDot({ activity }: { activity: InvestigationActivity }) {
  if (activity === "idle") return null;
  return (
    <span
      aria-hidden="true"
      className={cn(
        "size-1.5 rounded-full animate-in fade-in-0 zoom-in-50 duration-300",
        ACTIVITY_DOT[activity].className,
      )}
    />
  );
}

/** Inverted corner joining the tab's side border to the card's top border; plain CSS borders so both snap to the same pixels. */
function NotchCorner({ side, demo }: { side: "left" | "right"; demo: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "pointer-events-none absolute bottom-0 size-3 overflow-hidden",
        side === "left" ? "-left-3" : "-right-3",
      )}
    >
      <span className={frameCorner({ session: demo ? "demo" : "live", side })} />
    </span>
  );
}

export function LensModeSwitch({
  activity,
  demo,
  workers,
  setup,
}: {
  activity: InvestigationActivity;
  demo: boolean;
  workers: LensList["workers"] | null;
  setup?: string;
}) {
  const connected = useWorkerConnected(workers);
  const settingsTitle = connected ? "Worker connected" : "Connect worker";
  const tabs = Object.entries(LENS_TABS).filter(([view]) => view !== "settings" || workers);
  return (
    <div className={frameTab({ session: demo ? "demo" : "live" })}>
      <NotchCorner side="left" demo={demo} />
      <NotchCorner side="right" demo={demo} />
      <TabsPrimitive.List aria-label="Lens" className="relative inline-flex h-9 items-center p-1">
        <TabsPrimitive.Indicator className="absolute top-1 bottom-1 left-(--active-tab-left) w-(--active-tab-width) rounded-full bg-muted transition-[left,width] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none" />
        {tabs.map(([view, label]) => {
          const Icon = MODE_ICONS[view as LensTab];
          const workerDisconnected = view === "settings" && workers !== null && !connected;
          return (
            <TabsPrimitive.Tab
              key={view}
              value={view}
              title={view === "settings" ? settingsTitle : undefined}
              aria-description={
                view === "investigations" && activity !== "idle" ? ACTIVITY_DOT[activity].label : undefined
              }
              className={cn(
                "relative z-raised inline-flex h-full items-center gap-2 rounded-full text-sm font-medium text-muted-foreground outline-none transition-colors duration-200 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 data-active:text-foreground",
                view === "settings" ? "px-2.5" : "px-4",
              )}
            >
              <span className="relative inline-flex">
                <Icon aria-hidden="true" className="size-4" />
                {workerDisconnected && <StatusDot state="error" className="absolute -top-0.5 -right-0.5 size-1.5" />}
              </span>
              <span className={cn(view === "settings" && "sr-only")}>{label}</span>
              {view === "investigations" && setup && (
                <span className="absolute -top-3 right-2 rounded-full border border-border bg-card px-1.5 py-0.5 text-xs leading-none font-medium text-muted-foreground animate-in fade-in-0 duration-200">
                  {setup}
                </span>
              )}
              {view === "investigations" && <ActivityDot activity={activity} />}
            </TabsPrimitive.Tab>
          );
        })}
      </TabsPrimitive.List>
    </div>
  );
}
