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

import { isLive, presetLabel, RANGE_PRESETS, type RelativeRange, type TimeWindow, timeWindow } from "./timeRange";

const RANGE_LABEL_FORMAT = "MMM D, h:mm A";

const fixedRangeLabel = (range: TimeWindow): string =>
  `${moment(range.startMs).format(RANGE_LABEL_FORMAT)} to ${moment(range.endMs).format(RANGE_LABEL_FORMAT)}`;

/** A zoom or a paused range names its actual bounds; a live range names its preset. */
const rangeLabel = (range: RelativeRange, zoom: TimeWindow | null): string => {
  if (zoom) return fixedRangeLabel(zoom);
  return range.anchorMs === null ? presetLabel(range.hours) : fixedRangeLabel(timeWindow(range, range.anchorMs));
};

const SEGMENT = "inline-flex h-full items-center gap-1.5 px-3 text-sm outline-none focus-visible:bg-accent";

interface TimeRangeControlsProps {
  range: RelativeRange;
  zoom: TimeWindow | null;
  onHoursChange: (hours: number) => void;
  showLive?: boolean;
  onLiveChange: (live: boolean) => void;
}

/** Joined control group: the time range (opens presets) and Live. */
export function TimeRangeControls({ range, zoom, onHoursChange, showLive = true, onLiveChange }: TimeRangeControlsProps) {
  const live = isLive(range);
  return (
    <div className="flex items-stretch divide-x divide-border border-l border-border">
      <DropdownMenu>
        <DropdownMenuTrigger
          aria-label="Time range"
          className={cn(SEGMENT, "text-foreground hover:bg-muted/60")}
          data-testid="time-range-trigger"
        >
          <span className="tabular-nums">{rangeLabel(range, zoom)}</span>
          <ChevronDown className="size-3.5 text-muted-foreground" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-auto min-w-44">
          <DropdownMenuRadioGroup
            value={String(range.hours)}
            onValueChange={(value: string) => onHoursChange(Number(value))}
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
