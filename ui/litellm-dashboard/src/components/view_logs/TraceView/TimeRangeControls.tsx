"use client";

import { ChevronDown, Pause, Play, RotateCcw } from "lucide-react";
import moment from "moment";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

import type { TimeWindow } from "./TracesTimeline";

export const RANGE_PRESETS = [
  { hours: 1, label: "Last hour" },
  { hours: 6, label: "Last 6 hours" },
  { hours: 24, label: "Last 24 hours" },
  { hours: 168, label: "Last 7 days" },
  { hours: 720, label: "Last 30 days" },
] as const;

const RANGE_LABEL_FORMAT = "MMM D, h:mm A";

export const rangeLabel = (range: TimeWindow): string =>
  `${moment(range.startMs).format(RANGE_LABEL_FORMAT)} to ${moment(range.endMs).format(RANGE_LABEL_FORMAT)}`;

const SEGMENT = "inline-flex h-7 items-center gap-1.5 px-2.5 text-[13px] outline-none focus-visible:bg-accent";

interface TimeRangeControlsProps {
  range: TimeWindow;
  rangeHours: number;
  onRangeHoursChange: (hours: number) => void;
  live: boolean;
  showLive?: boolean;
  onLiveChange: (live: boolean) => void;
  onRefresh: () => void;
  refreshing: boolean;
}

/** Joined control group: refresh, the actual time range (opens presets), and Live. */
export function TimeRangeControls({
  range,
  rangeHours,
  onRangeHoursChange,
  live,
  showLive = true,
  onLiveChange,
  onRefresh,
  refreshing,
}: TimeRangeControlsProps) {
  return (
    <div className="flex items-center gap-1.5">
      <button
        type="button"
        onClick={onRefresh}
        aria-label="Refresh"
        title="Refresh"
        aria-busy={refreshing}
        className="inline-flex size-7 items-center justify-center rounded-md border border-border text-muted-foreground hover:text-foreground hover:bg-muted/60"
      >
        <RotateCcw className={cn("size-3.5", refreshing && "animate-spin")} />
      </button>
      <div className="flex items-center divide-x divide-border overflow-hidden rounded-md border border-border bg-card">
        <DropdownMenu>
          <DropdownMenuTrigger
            aria-label="Time range"
            className={cn(SEGMENT, "text-foreground hover:bg-muted/60")}
            data-testid="time-range-trigger"
          >
            <span className="tabular-nums">{rangeLabel(range)}</span>
            <ChevronDown className="size-3.5 text-muted-foreground" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-auto min-w-44">
            <DropdownMenuRadioGroup
              value={String(rangeHours)}
              onValueChange={(value: string) => onRangeHoursChange(Number(value))}
            >
              {RANGE_PRESETS.map((preset) => (
                <DropdownMenuRadioItem key={preset.hours} value={String(preset.hours)} className="text-[13px]">
                  {preset.label}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
        {showLive && (
          <button
            type="button"
            aria-pressed={live}
            onClick={() => onLiveChange(!live)}
            className={cn(
              SEGMENT,
              live ? "bg-info/10 text-info hover:bg-info/15" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {live ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            Live
          </button>
        )}
      </div>
    </div>
  );
}
