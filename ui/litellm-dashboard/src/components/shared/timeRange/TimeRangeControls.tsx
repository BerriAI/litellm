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
export function TimeRangeControls({
  range,
  zoom,
  onHoursChange,
  showLive = true,
  onLiveChange,
}: TimeRangeControlsProps) {
  const live = isLive(range);
  const label = rangeLabel(range, zoom);
  return (
    <div className="flex min-w-0 max-w-full items-stretch border-l border-border">
      <DropdownMenu>
        <DropdownMenuTrigger
          aria-label="Time range"
          className={cn(SEGMENT, "min-w-0 justify-between text-foreground hover:bg-muted/60 sm:min-w-40")}
          title={label}
          data-testid="time-range-trigger"
        >
          <span className="truncate tabular-nums">{label}</span>
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" sideOffset={1} className="rounded-t-none">
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
            "shrink-0 border-l border-border",
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
