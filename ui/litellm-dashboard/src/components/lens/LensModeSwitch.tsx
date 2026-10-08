"use client";

import { Tabs as TabsPrimitive } from "@base-ui/react/tabs";
import { Settings } from "lucide-react";
import { StatusDot } from "@/components/shared/StatusDot";
import { cn } from "@/lib/cva.config";
import type { InvestigationActivity } from "./model/status";
import { useWorkerConnected } from "./hooks/useWorkerConnected";
import type { LensList } from "./model/types";
import { LENS_TABS } from "./route";

const ACTIVITY_LABEL = { running: "An investigation is running", queued: "An investigation is queued" };

export function LensModeSwitch({
  activity,
  workers,
}: {
  activity: InvestigationActivity;
  workers: LensList["workers"] | null;
}) {
  const connected = useWorkerConnected(workers);
  const settingsTitle = connected ? "Worker connected" : "Connect worker";
  const tabs = Object.entries(LENS_TABS).filter(([view]) => view !== "settings" || workers);
  return (
    <TabsPrimitive.List aria-label="Lens" className="relative inline-flex h-8 max-w-full items-center gap-1">
      <TabsPrimitive.Indicator className="absolute -bottom-[9px] left-(--active-tab-left) h-0.5 w-(--active-tab-width) rounded-full bg-foreground transition-[left,width] duration-200 motion-reduce:transition-none" />
      {tabs.map(([view, label]) => (
        <TabsPrimitive.Tab
          key={view}
          value={view}
          title={view === "settings" ? settingsTitle : undefined}
          aria-description={view === "investigations" && activity !== "idle" ? ACTIVITY_LABEL[activity] : undefined}
          className={cn(
            "relative z-raised inline-flex h-full items-center gap-1.5 rounded-md text-sm font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 data-active:text-foreground",
            view === "settings" ? "px-2" : "px-2.5",
          )}
        >
          {view === "settings" ? (
            <span className="relative inline-flex">
              <Settings aria-hidden="true" className="size-3.5" />
              {!connected && <StatusDot state="error" className="absolute -top-0.5 -right-0.5 size-1.5" />}
              <span className="sr-only">{label}</span>
            </span>
          ) : (
            label
          )}
          {view === "investigations" && activity !== "idle" && (
            <span
              aria-hidden="true"
              className={cn(
                "size-1.5 rounded-full",
                activity === "running" ? "bg-info motion-safe:animate-pulse" : "bg-muted-foreground/60",
              )}
            />
          )}
        </TabsPrimitive.Tab>
      ))}
    </TabsPrimitive.List>
  );
}
