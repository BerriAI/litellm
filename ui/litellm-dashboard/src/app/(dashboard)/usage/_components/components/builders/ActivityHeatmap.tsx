"use client";

import { Panel } from "../overview/Primitives";

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] as const;

const formatDay = (date: string) => {
  const parsed = new Date(`${date}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? date : parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

const captionForRange = (start: string, end: string) => {
  const startLabel = formatDay(start);
  const endLabel = formatDay(end);
  const startMonth = startLabel.split(" ")[0];
  const endMonth = endLabel.split(" ")[0];
  return startMonth === endMonth ? `${startLabel} to ${endLabel.split(" ")[1]}` : `${startLabel} to ${endLabel}`;
};

export function ActivityHeatmap({
  heat,
  sampleStart,
  sampleEnd,
}: {
  heat: readonly (readonly number[])[];
  sampleStart: string;
  sampleEnd: string;
}) {
  const max = Math.max(1, ...heat.flat());

  return (
    <Panel title="When they work">
      <div className="overflow-x-auto">
        <div className="min-w-[34rem]">
          <div className="grid grid-cols-[2.5rem_repeat(24,minmax(0,1fr))] gap-1 text-[9px] text-muted-foreground">
            <span />
            {Array.from({ length: 24 }, (_, hour) => (
              <span key={hour} className="text-center">
                {hour % 6 === 0 ? hour : ""}
              </span>
            ))}
          </div>
          <div className="mt-1 grid gap-1">
            {heat.slice(0, 7).map((row, rowIndex) => (
              <div key={WEEKDAYS[rowIndex]} className="grid grid-cols-[2.5rem_repeat(24,minmax(0,1fr))] items-center gap-1">
                <span className="text-[10px] text-muted-foreground">{WEEKDAYS[rowIndex]}</span>
                {Array.from({ length: 24 }, (_, hour) => {
                  const value = row[hour] ?? 0;
                  const intensity = value > 0 ? 0.2 + (value / max) * 0.8 : 1;
                  return (
                    <span
                      key={hour}
                      title={`${WEEKDAYS[rowIndex]} ${hour}:00, ${value.toLocaleString()} requests`}
                      className="aspect-square rounded-[2px]"
                      style={{
                        backgroundColor: value > 0 ? "var(--chart-1)" : "var(--muted)",
                        opacity: intensity,
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">Requests by hour, PT, {captionForRange(sampleStart, sampleEnd)}</p>
    </Panel>
  );
}
