"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { roiRequest } from "./api";
import { CategoryDrawer } from "./CategoryDrawer";
import { CategorySetup } from "./CategorySetup";
import { roiTotals, selectedSessionDetail, type Example, type Report, type Status } from "./data";
import { SemanticSpendOverview } from "./SemanticSpendOverview";

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
  const [area, setArea] = useState<string | null>(null);
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
      roiRequest<Example[]>(token, `/examples?${params}&session_key=${encodeURIComponent(selectedId ?? "")}`),
    enabled: active && selectedId !== null,
  });
  const { total } = roiTotals(report.data);
  const selected = selectedSessionDetail(report.data, selectedId);
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
    <div className="space-y-6 rounded-xl bg-muted/20 p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Semantic Spend</h2>
          <p className="text-sm text-muted-foreground">
            What your team’s model spend was used for, grouped by session.
          </p>
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
          {report.data && (
            <SemanticSpendOverview
              report={report.data}
              area={area}
              onArea={setArea}
              onSession={(session) => setSelectedId(session.session_key)}
            />
          )}
          <p className="text-xs text-muted-foreground">
            Classification updates in the background. Discovery samples remain unclassified. ROI analysis calls are
            excluded from task spend. This view attributes cost; it does not estimate business returns.
          </p>
        </>
      )}
      <CategoryDrawer selected={selected} total={total} examples={examples} onClose={() => setSelectedId(null)} />
    </div>
  );
}
