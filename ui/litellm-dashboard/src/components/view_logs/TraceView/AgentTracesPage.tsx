"use client";

import moment from "moment";
import { useMemo, useState } from "react";

import { AgentTracesSection } from "./AgentTracesSection";

const DEFAULT_RANGE_HOURS = 24;
const TIME_FORMAT = "YYYY-MM-DDTHH:mm";

export default function AgentTracesPage({
  accessToken,
  isActive = true,
  readOnly = false,
  canMintTracingKey = false,
}: {
  accessToken: string;
  isActive?: boolean;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
}) {
  const [rangeHours, setRangeHours] = useState(DEFAULT_RANGE_HOURS);
  const [live, setLive] = useState(true);
  const [anchor, setAnchor] = useState(() => moment());
  const { startTime, endTime } = useMemo(
    () => ({
      startTime: anchor.clone().subtract(rangeHours, "hours").format(TIME_FORMAT),
      endTime: anchor.format(TIME_FORMAT),
    }),
    [anchor, rangeHours],
  );

  const changeRange = (hours: number) => {
    setRangeHours(hours);
    setAnchor(moment());
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col pt-3">
      <AgentTracesSection
        accessToken={accessToken}
        isActive={isActive}
        startTime={startTime}
        endTime={endTime}
        isCustomDate={false}
        isLiveTail={live}
        readOnly={readOnly}
        canMintTracingKey={canMintTracingKey}
        timeControls={{ rangeHours, onRangeHoursChange: changeRange, onLiveChange: setLive }}
      />
    </div>
  );
}
