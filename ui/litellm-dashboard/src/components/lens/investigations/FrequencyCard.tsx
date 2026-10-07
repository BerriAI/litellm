"use client";

import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { dayLabel, type Frequency, percentLabel } from "../model/frequency";

const AFFECTED = "var(--finding-affected)";
const UNAFFECTED = "var(--finding-unaffected)";
const TICK = { fontSize: 12, fill: "var(--muted-foreground)" };

function Legend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <span aria-hidden="true" className="size-2 rounded-full bg-finding-affected" />
        Affected
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden="true" className="size-2 rounded-full bg-finding-unaffected" />
        Unaffected
      </span>
    </div>
  );
}

export function FrequencyCard({ frequency }: { frequency: Frequency }) {
  const percent = percentLabel(frequency.affected, frequency.total);
  const data = frequency.days.map((d) => ({ ...d, label: dayLabel(d.day) }));
  return (
    <section aria-label="Frequency" className="space-y-3 rounded-lg border border-border/60 bg-muted/40 p-4">
      <div aria-live="polite">
        <h3 className="mb-1 text-sm font-medium">Frequency</h3>
        <p className="min-h-7 text-lg font-medium tabular-nums">
          {percent ?? "—"}
          <span className="text-sm font-normal text-muted-foreground">
            {" "}
            {frequency.affected} of {frequency.total} traces affected
          </span>
        </p>
      </div>
      {data.length > 0 && (
        <div className="h-60 md:h-72" data-testid="frequency-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              margin={{ top: 8, right: 4, bottom: 0, left: -16 }}
              barCategoryGap="12%"
              maxBarSize={56}
            >
              <XAxis
                dataKey="label"
                tick={TICK}
                tickLine={false}
                axisLine={{ stroke: "var(--border)" }}
                minTickGap={24}
              />
              <YAxis allowDecimals={false} tick={TICK} tickLine={false} axisLine={false} width={40} />
              <Tooltip
                cursor={{ fill: "var(--muted)", opacity: 0.6 }}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="affected" name="Affected" stackId="traces" fill={AFFECTED} isAnimationActive={false} />
              <Bar
                dataKey="unaffected"
                name="Unaffected"
                stackId="traces"
                fill={UNAFFECTED}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <Legend />
      <p className="text-xs text-muted-foreground">Traces sampled by the investigations that reported this finding</p>
    </section>
  );
}
