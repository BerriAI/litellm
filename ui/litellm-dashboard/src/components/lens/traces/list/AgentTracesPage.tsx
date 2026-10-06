"use client";

import moment from "moment";
import { useMemo, useState } from "react";

import { AgentTracesSection } from "./AgentTracesSection";
import { useTracesLive } from "../api";
import { useRangeHoursRouting } from "../routing";

const TIME_FORMAT = "YYYY-MM-DDTHH:mm:ss";

export default function AgentTracesPage({
  accessToken,
  isActive = true,
  readOnly = false,
  canMintTracingKey = false,
  canViewFindings = true,
}: {
  accessToken: string;
  isActive?: boolean;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
  canViewFindings?: boolean;
}) {
  const sourceLive = useTracesLive();
  const [rangeHours, setRangeHours] = useRangeHoursRouting();
  const [live, setLive] = useState(true);
  const [anchor, setAnchor] = useState(() => moment());
  const { startTime, endTime } = useMemo(
    () => ({
      startTime: anchor.clone().subtract(rangeHours, "hours").format(TIME_FORMAT),
      endTime: anchor.format(TIME_FORMAT),
    }),
    [anchor, rangeHours],
  );

  const isLiveTail = live && sourceLive;

  const changeRange = (hours: number) => {
    setRangeHours(hours);
    setAnchor(moment());
  };

  const changeLive = (next: boolean) => {
    setLive(next);
    setAnchor(moment());
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <AgentTracesSection
        accessToken={accessToken}
        isActive={isActive}
        startTime={startTime}
        endTime={endTime}
        isCustomDate={!isLiveTail}
        isLiveTail={isLiveTail}
        readOnly={readOnly}
        canMintTracingKey={canMintTracingKey}
        canViewFindings={canViewFindings}
        timeControls={{ rangeHours, onRangeHoursChange: changeRange, onLiveChange: changeLive }}
      />
    </div>
  );
}
