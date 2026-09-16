"use client";

import { CalendarDays } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";

import { QUICK_SELECT_OPTIONS, type QuickSelectPresetId } from "./constants";
import { getTimeRangeDisplay } from "./logs_utils";
import { CUSTOM_RANGE, type LogsTimeRange } from "./useLogsTimeRange";

interface LogsTableToolbarProps {
  timeRange: LogsTimeRange;
  onPresetSelect: (preset: QuickSelectPresetId) => void;
  onCustomRangeToggle: () => void;
  onStartTimeChange: (value: string) => void;
  onEndTimeChange: (value: string) => void;
  isLiveTail: boolean;
  onIsLiveTailChange: (value: boolean) => void;
  excludeInternalHealthChecks: boolean;
  onExcludeInternalHealthChecksChange: (value: boolean) => void;
  onResetFilters: () => void;
}

export function LogsTableToolbar({
  timeRange,
  onPresetSelect,
  onCustomRangeToggle,
  onStartTimeChange,
  onEndTimeChange,
  isLiveTail,
  onIsLiveTailChange,
  excludeInternalHealthChecks,
  onExcludeInternalHealthChecksChange,
  onResetFilters,
}: LogsTableToolbarProps) {
  const [quickSelectOpen, setQuickSelectOpen] = useState(false);
  const { range, startTime, endTime } = timeRange;
  const isCustomDate = range === CUSTOM_RANGE;

  const displayLabel = isCustomDate
    ? getTimeRangeDisplay(true, startTime, endTime)
    : QUICK_SELECT_OPTIONS.find((option) => option.id === range)?.label;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Popover open={quickSelectOpen} onOpenChange={setQuickSelectOpen}>
        <PopoverTrigger
          render={
            <Button variant="outline" size="sm" className="gap-2">
              <CalendarDays className="size-4" />
              {displayLabel}
            </Button>
          }
        />
        <PopoverContent align="start" className="w-64 p-2">
          <div className="space-y-1">
            {QUICK_SELECT_OPTIONS.map((option) => (
              <Button
                key={option.id}
                variant="ghost"
                className="w-full justify-start font-normal"
                onClick={() => {
                  onPresetSelect(option.id);
                  setQuickSelectOpen(false);
                }}
              >
                {option.label}
              </Button>
            ))}
            <div className="my-2 border-t" />
            <Button variant="ghost" className="w-full justify-start font-normal" onClick={onCustomRangeToggle}>
              Custom Range
            </Button>
          </div>
        </PopoverContent>
      </Popover>

      {isCustomDate && (
        <div className="flex items-center gap-2">
          <Input
            type="datetime-local"
            className="w-auto"
            aria-label="Start time"
            value={startTime}
            onChange={(event) => onStartTimeChange(event.target.value)}
          />
          <span className="text-sm text-muted-foreground">to</span>
          <Input
            type="datetime-local"
            className="w-auto"
            aria-label="End time"
            value={endTime}
            onChange={(event) => onEndTimeChange(event.target.value)}
          />
        </div>
      )}

      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Live Tail</span>
        <Switch checked={isLiveTail} onCheckedChange={onIsLiveTailChange} aria-label="Live Tail" />
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Hide Health Checks</span>
        <Switch
          checked={excludeInternalHealthChecks}
          onCheckedChange={onExcludeInternalHealthChecksChange}
          aria-label="Hide Health Checks"
        />
      </div>

      <Button variant="outline" size="sm" onClick={onResetFilters}>
        Reset Filters
      </Button>
    </div>
  );
}

export function LiveTailBanner({ onStop }: { onStop: () => void }) {
  return (
    <div className="mb-4 flex items-center justify-between rounded-md border border-success/20 bg-success/10 px-4 py-2">
      <span className="text-sm text-success">Auto-refreshing every 15 seconds</span>
      <button type="button" onClick={onStop} className="text-sm text-success hover:text-success/80">
        Stop
      </button>
    </div>
  );
}
