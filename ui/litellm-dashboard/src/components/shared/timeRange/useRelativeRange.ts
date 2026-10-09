import { useMemo, useState } from "react";

import { useRangeHoursRouting } from "./routing";
import type { RelativeRange } from "./timeRange";

export interface RelativeRangeState {
  readonly range: RelativeRange;
  readonly setHours: (hours: number) => void;
  readonly setLive: (live: boolean) => void;
}

/** The preset comes from the URL; live is page state. Any change re-anchors a paused range to now. */
export function useRelativeRange(canGoLive: boolean): RelativeRangeState {
  const [hours, setRangeHours] = useRangeHoursRouting();
  const [live, setLiveState] = useState(true);
  const [anchorMs, setAnchorMs] = useState(() => Date.now());
  const rolling = live && canGoLive;
  const range = useMemo(() => ({ hours, anchorMs: rolling ? null : anchorMs }), [hours, rolling, anchorMs]);
  return {
    range,
    setHours: (next) => {
      setRangeHours(next);
      setAnchorMs(Date.now());
    },
    setLive: (next) => {
      setLiveState(next);
      setAnchorMs(Date.now());
    },
  };
}
