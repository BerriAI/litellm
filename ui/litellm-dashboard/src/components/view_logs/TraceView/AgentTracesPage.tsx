"use client";
import { useLensDemo } from "@/components/lens/LensDemoContext";

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
  onDemo,
}: {
  accessToken: string;
  isActive?: boolean;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
  onDemo?: () => void;
}) {
  const demo = useLensDemo();
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
    <div className="flex min-h-0 flex-1 flex-col">
      <AgentTracesSection
        accessToken={accessToken}
        isActive={isActive}
        startTime={startTime}
        endTime={endTime}
        isCustomDate={false}
        isLiveTail={live && !demo}
        readOnly={readOnly}
        canMintTracingKey={canMintTracingKey}
        onDemo={onDemo}
        timeControls={{ rangeHours, onRangeHoursChange: changeRange, onLiveChange: setLive }}
      />
    </div>
  );
}
