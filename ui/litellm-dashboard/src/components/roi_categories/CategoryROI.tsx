"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { StackedUsageChart } from "@/components/shared/charts";
import { stackedUsageColor } from "@/components/shared/charts/StackedUsageChart";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { roiRequest } from "./api";
import { CategoryDrawer } from "./CategoryDrawer";
import { CategorySetup } from "./CategorySetup";
import { categoryChart, roiTotals, money, type Example, type Report, type Status } from "./data";

export default function CategoryROI({
  token,
  start,
  end,
  user = "",
}: {
  token: string;
  start: string;
  end: string;
  user?: string;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const status = useQuery({
    queryKey: ["roi-categories-status", token],
    queryFn: () => roiRequest<Status>(token, "/status"),
    refetchInterval: 10000,
  });
  const active = status.data?.settings?.phase === "active";
  const params = `start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&model=${encodeURIComponent(model)}&user=${encodeURIComponent(user)}`;
  const reportQuery = {
    queryKey: ["roi-categories-report", token, params, status.data?.settings?.version],
    queryFn: () => roiRequest<Report>(token, `/report?${params}`),
    enabled: active,
    refetchInterval: 15000,
  };
  const report = useQuery(reportQuery);
  const examples = useQuery({
    queryKey: ["roi-categories-examples", token, params, selectedId],
    queryFn: () =>
      roiRequest<Example[]>(token, `/examples?${params}&category_id=${encodeURIComponent(selectedId ?? "")}`),
    enabled: active && selectedId !== null,
  });
  const { rows, total, requests, categorized, unknownCosts, coverage, unclassifiedSpend } = roiTotals(report.data);
  const selected = rows.find((row) => row.id === selectedId);
  const error = status.error ?? report.error;

  if (status.isPending)
    return (
      <p role="status" className="p-6 text-sm text-muted-foreground">
        Loading ROI…
      </p>
    );
  if (status.error)
    return (
      <p role="alert" className="p-6 text-destructive">
        {status.error.message}
      </p>
    );
  if (!status.data?.configured)
    return (
      <div className="rounded-lg border p-6">
        <h3 className="font-medium">Connect ClickHouse to explore ROI</h3>
        <p className="mt-2 text-sm text-muted-foreground">
          Enable ClickHouse spend logging on your gateway, then return here to discover task categories.
        </p>
        <Button className="mt-4" variant="outline" onClick={() => void status.refetch()}>
          Check connection
        </Button>
      </div>
    );
  if (!active)
    return (
      <CategorySetup
        key={status.data.settings?.phase ?? "new"}
        token={token}
        settings={status.data.settings}
        onSaved={() => void status.refetch()}
      />
    );

  return (
    <div className="space-y-6 pt-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">Spend by task</h2>
          <p className="text-sm text-muted-foreground">See what your LLM traffic is doing and where its cost goes.</p>
        </div>
        <Input
          aria-label="Filter ROI by model"
          className="max-w-xs"
          placeholder="Filter by exact model name"
          value={model}
          onChange={(event) => setModel(event.target.value)}
        />
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error.message}
        </p>
      )}
      {report.isPending ? (
        <p role="status">Loading category spend…</p>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            {[
              { label: "Task spend", value: money(total), detail: "Recorded gateway costs" },
              {
                label: "Categorized requests",
                value: `${coverage}%`,
                detail: `${categorized.toLocaleString()} of ${requests.toLocaleString()} requests`,
              },
              {
                label: "Unclassified spend",
                value: money(unclassifiedSpend),
                detail: "Pending, unavailable, or unmatched",
              },
            ].map((card) => (
              <div key={card.label} className="rounded-lg border bg-card p-5">
                <p className="text-sm text-muted-foreground">{card.label}</p>
                <p className="mt-2 text-2xl font-semibold tabular-nums">{card.value}</p>
                <p className="mt-1 text-xs text-muted-foreground">{card.detail}</p>
              </div>
            ))}
          </div>
          {unknownCosts > 0 && (
            <p className="text-sm text-muted-foreground">
              Cost is unavailable for {unknownCosts} requests. Spend totals include known costs only.
            </p>
          )}
          <section className="rounded-lg border bg-card p-5">
            <h3 className="font-medium">Category spend over time</h3>
            <p className="mb-5 text-sm text-muted-foreground">Daily spend, stacked by task category · UTC</p>
            {report.data && requests > 0 ? (
              <StackedUsageChart
                data={categoryChart(report.data)}
                series={rows.map((row) => row.id)}
                labels={rows.map((row) => row.name)}
                colors={rows.map((row, index) => (row.id === "unclassified" ? "#94a3b8" : stackedUsageColor(index)))}
                xKey="day"
                format={money}
              />
            ) : (
              <p className="py-12 text-center text-sm text-muted-foreground">
                No traffic in this date range. Send requests through the gateway to see category spend.
              </p>
            )}
          </section>
          <section className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="p-4">Category</th>
                  <th className="p-4 text-right">Spend</th>
                  <th className="p-4 text-right">Share</th>
                  <th className="p-4 text-right">Requests</th>
                  <th className="p-4">Top model</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={row.id} className="border-b last:border-0 hover:bg-muted/40">
                    <td className="p-4">
                      <button
                        className="flex items-center gap-2 font-medium hover:underline"
                        onClick={() => setSelectedId(row.id)}
                      >
                        <span
                          className="size-2.5 rounded-full"
                          style={{ background: row.id === "unclassified" ? "#94a3b8" : stackedUsageColor(index) }}
                        />
                        {row.name}
                      </button>
                    </td>
                    <td className="p-4 text-right tabular-nums">{money(row.spend)}</td>
                    <td className="p-4 text-right tabular-nums">
                      {total ? ((100 * row.spend) / total).toFixed(1) : "0"}%
                    </td>
                    <td className="p-4 text-right tabular-nums">{row.requests.toLocaleString()}</td>
                    <td className="p-4 text-muted-foreground">{row.models[0]?.model ?? "None"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <p className="text-xs text-muted-foreground">
            Classification updates in the background. Discovery samples remain unclassified. Analysis calls are excluded
            from task spend. This view attributes cost; it does not estimate business returns.
          </p>
        </>
      )}
      <CategoryDrawer selected={selected} total={total} examples={examples} onClose={() => setSelectedId(null)} />
    </div>
  );
}
