"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { dayLabel, type Frequency, percentLabel } from "../model/frequency";

const AFFECTED = "var(--finding-affected)";
const UNAFFECTED = "var(--finding-unaffected)";
const TICK = { fontSize: 11, fill: "var(--muted-foreground)" };
const TOP_RADIUS: [number, number, number, number] = [3, 3, 0, 0];

function Legend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <span aria-hidden="true" className="size-2 rounded-sm bg-finding-affected" />
        Affected
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden="true" className="size-2 rounded-sm bg-finding-unaffected" />
        Unaffected
      </span>
    </div>
  );
}

export function FrequencyCard({ frequency }: { frequency: Frequency }) {
  const percent = percentLabel(frequency.affected, frequency.total);
  const data = frequency.days.map((d) => ({ ...d, label: dayLabel(d.day) }));
  const range = data.length > 0 ? `${data[0].label} – ${data[data.length - 1].label}` : null;
  return (
    <section
      aria-label="Frequency"
      className="space-y-4 rounded-xl border border-transparent bg-muted/40 p-4 shadow-finding-ring"
    >
      <div aria-live="polite" className="flex items-start justify-between gap-4">
        <div>
          <h3 className="mb-1 text-sm font-medium">Frequency</h3>
          <p className="text-2xl leading-tight font-semibold tracking-tight tabular-nums">
            {percent ?? "—"}
            <span className="ml-1.5 text-sm font-normal tracking-normal text-muted-foreground">
              {frequency.affected} of {frequency.total} traces affected
            </span>
          </p>
        </div>
        {range && <span className="pt-0.5 text-xs whitespace-nowrap text-muted-foreground">{range}</span>}
      </div>
      {data.length > 0 && (
        <div className="h-56 md:h-64" data-testid="frequency-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              margin={{ top: 4, right: 0, bottom: 0, left: -20 }}
              barCategoryGap="18%"
              maxBarSize={44}
            >
              <CartesianGrid vertical={false} stroke="var(--border)" strokeOpacity={0.6} />
              <XAxis dataKey="label" tick={TICK} tickLine={false} axisLine={false} minTickGap={28} tickMargin={8} />
              <YAxis allowDecimals={false} tick={TICK} tickLine={false} axisLine={false} width={40} tickCount={4} />
              <Tooltip
                cursor={{ fill: "var(--muted)", opacity: 0.7 }}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  fontSize: 12,
                  boxShadow: "var(--finding-ring)",
                }}
              />
              <Bar dataKey="affected" name="Affected" stackId="traces" fill={AFFECTED} isAnimationActive={false} />
              <Bar
                dataKey="unaffected"
                name="Unaffected"
                stackId="traces"
                fill={UNAFFECTED}
                radius={TOP_RADIUS}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Legend />
        <p className="text-xs text-muted-foreground">Share of traces sampled by the reporting investigations</p>
      </div>
    </section>
  );
}
