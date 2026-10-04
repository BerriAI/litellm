"use client";

import { Tabs as TabsPrimitive } from "@base-ui/react/tabs";
import { Activity, ScanSearch, Settings } from "lucide-react";
import { StatusDot } from "@/components/shared/StatusDot";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";
import { workerConnected, type InvestigationActivity } from "./model/status";
import type { LensList } from "./model/types";
import { LENS_TABS, type LensTab } from "./route";

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

const FRAME = {
  live: {
    card: "border border-foreground/15",
    tab: "-mb-px border-x border-t border-foreground/15",
    corner: { left: "border-r border-b", right: "border-l border-b", tone: "border-foreground/15" },
  },
  demo: {
    card: "border-2 border-info",
    tab: "-mb-0.5 border-x-2 border-t-2 border-info",
    corner: { left: "border-r-2 border-b-2", right: "border-l-2 border-b-2", tone: "border-info" },
  },
} as const;

export const frameOf = (demo: boolean) => FRAME[demo ? "demo" : "live"];

/** Inverted corner joining the tab's side border to the card's top border; plain CSS borders so both snap to the same pixels. */
function NotchCorner({ side, demo }: { side: "left" | "right"; demo: boolean }) {
  const { corner } = frameOf(demo);
  return (
    <span
      aria-hidden="true"
      className={cn(
        "pointer-events-none absolute bottom-0 size-3 overflow-hidden",
        side === "left" ? "-left-3" : "-right-3",
      )}
    >
      <span
        className={cn(
          "block size-full shadow-[0_0_0_12px_var(--card)]",
          side === "left" ? "rounded-br-xl" : "rounded-bl-xl",
          corner[side],
          corner.tone,
        )}
      />
    </span>
  );
}

const HEARTBEAT_TICK_MS = 10000;

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
  const now = useNow(HEARTBEAT_TICK_MS);
  const connected = workers?.some((candidate) => workerConnected(candidate, now)) ?? false;
  const settingsTitle = connected ? "Worker connected" : "Connect worker";
  const tabs = Object.entries(LENS_TABS).filter(([view]) => view !== "settings" || workers);
  return (
    <div className={cn("relative z-raised rounded-t-2xl bg-card px-1.5 pt-1.5 pb-[7px]", frameOf(demo).tab)}>
      <NotchCorner side="left" demo={demo} />
      <NotchCorner side="right" demo={demo} />
      <TabsPrimitive.List
        aria-label="Lens"
        className="relative inline-flex h-9 items-center rounded-full bg-muted/70 p-1"
      >
        <TabsPrimitive.Indicator className="absolute top-1 bottom-1 left-(--active-tab-left) w-(--active-tab-width) rounded-full bg-background shadow-sm ring-1 ring-border transition-[left,width] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none" />
        {tabs.map(([view, label]) => {
          const Icon = MODE_ICONS[view as LensTab];
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
                {view === "settings" && workers && (
                  <StatusDot
                    state={connected ? "ok" : "warn"}
                    className="absolute -top-0.5 -right-0.5 size-1.5 ring-2 ring-muted"
                  />
                )}
              </span>
              <span className={cn(view === "settings" && "sr-only")}>{label}</span>
              {view === "investigations" && setup && (
                <span className="rounded-full bg-muted px-1.5 py-0.5 text-xs leading-none font-medium text-muted-foreground animate-in fade-in-0 duration-200">
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
