"use client";

import { ChevronDown, Pause, Play } from "lucide-react";
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

const fixedRangeLabel = (range: TimeWindow): string =>
  `${moment(range.startMs).format(RANGE_LABEL_FORMAT)} to ${moment(range.endMs).format(RANGE_LABEL_FORMAT)}`;

const presetLabel = (hours: number): string =>
  RANGE_PRESETS.find((preset) => preset.hours === hours)?.label ?? `Last ${hours} hours`;

const SEGMENT = "inline-flex h-7 items-center gap-1.5 px-2.5 text-sm outline-none focus-visible:bg-accent";

interface TimeRangeControlsProps {
  fixedRange: TimeWindow | null;
  rangeHours: number;
  onRangeHoursChange: (hours: number) => void;
  live: boolean;
  showLive?: boolean;
  onLiveChange: (live: boolean) => void;
}

/** Joined control group: the time range (opens presets) and Live. */
export function TimeRangeControls({
  fixedRange,
  rangeHours,
  onRangeHoursChange,
  live,
  showLive = true,
  onLiveChange,
}: TimeRangeControlsProps) {
  return (
    <div className="flex items-center divide-x divide-border overflow-hidden rounded-lg border border-border bg-card">
      <DropdownMenu>
        <DropdownMenuTrigger
          aria-label="Time range"
          className={cn(SEGMENT, "text-foreground hover:bg-muted/60")}
          data-testid="time-range-trigger"
        >
          <span className="tabular-nums">{fixedRange ? fixedRangeLabel(fixedRange) : presetLabel(rangeHours)}</span>
          <ChevronDown className="size-3.5 text-muted-foreground" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-auto min-w-44">
          <DropdownMenuRadioGroup
            value={String(rangeHours)}
            onValueChange={(value: string) => onRangeHoursChange(Number(value))}
          >
            {RANGE_PRESETS.map((preset) => (
              <DropdownMenuRadioItem key={preset.hours} value={String(preset.hours)} className="text-sm">
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
  );
}
