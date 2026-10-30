import { ChevronRight } from "lucide-react";
import { StackedUsageChart } from "@/components/shared/charts";
import { areaColor, categoryChart, money, roiTotals, sessionTitle, type Report, type SessionSpend } from "./data";

const dayLabel = (value: string) =>
  new Date(`${value}T00:00:00Z`).toLocaleDateString("en-US", { weekday: "short", day: "numeric", timeZone: "UTC" });
const startedOptions: Intl.DateTimeFormatOptions = {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: "UTC",
};
const startedFormat = new Intl.DateTimeFormat("en-US", startedOptions);
const startedLabel = (value: string) => startedFormat.format(new Date(value));

export function SemanticSpendOverview({
  report,
  area,
  onArea,
  onSession,
}: {
  report: Report;
  area: string | null;
  onArea: (id: string | null) => void;
  onSession: (session: SessionSpend) => void;
}) {
  const { rows, total, requests, unknownCosts, coverage } = roiTotals(report);
  const sessions = report.sessions ?? [];
  const grouped = sessions.filter((session) => session.session_id);
  const average = grouped.length ? grouped.reduce((sum, session) => sum + session.spend, 0) / grouped.length : null;
  const visible = area ? sessions.filter((session) => session.category_ids.includes(area)) : sessions;
  const highest = Math.max(0, ...sessions.map((session) => session.spend));
  return (
    <div className="space-y-6">
      <section
        aria-label="Spend summary"
        className="grid overflow-hidden rounded-xl border bg-card md:grid-cols-[2fr_1fr_1fr_1fr]"
      >
        <div className="border-b p-6 md:border-r md:border-b-0">
          <p className="text-sm text-muted-foreground">Total spend</p>
          <p className="mt-3 text-4xl font-semibold tracking-tight tabular-nums">{money(total)}</p>
          <p className="mt-3 text-xs text-muted-foreground">
            {coverage}% of requests categorized · Known gateway costs
          </p>
        </div>
        {[
          { label: "Sessions", value: grouped.length.toLocaleString(), detail: "With a recorded session ID" },
          {
            label: "Requests",
            value: requests.toLocaleString(),
            detail: `${sessions.length - grouped.length} standalone requests`,
          },
          {
            label: "Avg per session",
            value: average === null ? "—" : money(average),
            detail: requests ? `${money(total / requests)} / request` : "No requests in this period",
          },
        ].map((metric) => (
          <div key={metric.label} className="border-b p-6 last:border-0 md:border-r md:border-b-0">
            <p className="text-sm text-muted-foreground">{metric.label}</p>
            <p className="mt-3 text-2xl font-semibold tabular-nums">{metric.value}</p>
            <p className="mt-2 text-xs text-muted-foreground">{metric.detail}</p>
          </div>
        ))}
      </section>
      {unknownCosts > 0 && (
        <p className="text-sm text-muted-foreground">
          Cost is unavailable for {unknownCosts} requests. Totals include known costs only.
        </p>
      )}
      <section className="rounded-xl border bg-card p-6">
        <div className="mb-7 flex flex-wrap items-start justify-between gap-5">
          <div>
            <h3 className="font-semibold">Spend by day</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Stacked by work area. Select an area to focus the session list.
            </p>
          </div>
          <div aria-label="Work area legend" className="flex flex-wrap gap-x-5 gap-y-2">
            {rows
              .filter((row) => row.requests > 0)
              .map((row) => (
                <button
                  key={row.id}
                  aria-pressed={area === row.id}
                  onClick={() => onArea(area === row.id ? null : row.id)}
                  className="flex items-center gap-2 rounded px-1 py-1 text-sm hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <span className="size-2.5 rounded-sm" style={{ background: areaColor(row.id, report.categories) }} />
                  <span className="text-muted-foreground">{row.name}</span>
                  <span className="font-medium tabular-nums">{money(row.spend)}</span>
                </button>
              ))}
          </div>
        </div>
        {requests > 0 ? (
          <StackedUsageChart
            data={categoryChart(report)}
            series={rows.map((row) => row.id)}
            labels={rows.map((row) => row.name)}
            colors={rows.map((row) => (area && area !== row.id ? "#dce1e8" : areaColor(row.id, report.categories)))}
            xKey="day"
            xLabel={dayLabel}
            format={money}
            className="h-72 w-full"
          />
        ) : (
          <p className="py-16 text-center text-sm text-muted-foreground">No traffic in this date range.</p>
        )}
      </section>
      <section className="overflow-hidden rounded-xl border bg-card">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b px-6 py-5">
          <h3 className="font-semibold">Sessions</h3>
          <div aria-label="Filter sessions by work area" className="flex flex-wrap gap-1 rounded-lg bg-muted/60 p-1">
            {[{ id: null, name: "All areas" }, ...report.categories, { id: "unclassified", name: "Unclassified" }].map(
              (item) => (
                <button
                  key={item.id ?? "all"}
                  aria-pressed={area === item.id}
                  onClick={() => onArea(item.id)}
                  className={`rounded-md px-3 py-1.5 text-sm ${area === item.id ? "bg-background font-medium shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                >
                  {item.name}
                </button>
              ),
            )}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="border-b bg-muted/25 text-left text-xs font-normal text-muted-foreground">
              <tr>
                <th className="px-6 py-3 font-normal">Session</th>
                <th className="px-4 py-3 font-normal">Started · UTC</th>
                <th className="px-4 py-3 text-right font-normal">Requests</th>
                <th className="px-6 py-3 text-right font-normal">Spend</th>
                <th className="w-10">
                  <span className="sr-only">Details</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {visible.map((session) => {
                const category = session.category_ids.length === 1 ? session.category_ids[0] : "mixed";
                const name =
                  category === "mixed"
                    ? "Mixed work areas"
                    : rows.find((row) => row.id === category)?.name ?? "Unclassified";
                const color = category === "mixed" ? "#64748b" : areaColor(category, report.categories);
                return (
                  <tr key={session.session_key} className="border-b last:border-0 hover:bg-muted/30">
                    <td className="max-w-md px-6 py-4">
                      <button className="flex w-full items-start gap-3 text-left" onClick={() => onSession(session)}>
                        <span className="mt-1.5 size-2.5 shrink-0 rounded-full" style={{ background: color }} />
                        <span className="min-w-0">
                          <span className="block truncate font-medium">{sessionTitle(session)}</span>
                          <span className="mt-1 block truncate text-xs text-muted-foreground">
                            {name} · {session.user || "Unknown user"}
                            {session.session_id ? "" : " · Standalone request"}
                          </span>
                        </span>
                      </button>
                    </td>
                    <td className="whitespace-nowrap px-4 py-4 text-muted-foreground">
                      {startedLabel(session.started_at)}
                    </td>
                    <td className="px-4 py-4 text-right tabular-nums text-muted-foreground">
                      {session.requests.toLocaleString()}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-end gap-5">
                        <span className="hidden h-1.5 w-24 overflow-hidden rounded-full bg-muted lg:block">
                          <span
                            className="block h-full rounded-full"
                            style={{ width: `${highest ? (100 * session.spend) / highest : 0}%`, background: color }}
                          />
                        </span>
                        <span className="min-w-20 text-right font-medium tabular-nums">{money(session.spend)}</span>
                      </div>
                    </td>
                    <td className="pr-4">
                      <button
                        aria-label={`View ${sessionTitle(session)}`}
                        onClick={() => onSession(session)}
                        className="p-1 text-muted-foreground"
                      >
                        <ChevronRight className="size-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {visible.length === 0 && (
          <p className="p-10 text-center text-sm text-muted-foreground">
            No sessions or standalone requests match this view.
          </p>
        )}
      </section>
    </div>
  );
}
