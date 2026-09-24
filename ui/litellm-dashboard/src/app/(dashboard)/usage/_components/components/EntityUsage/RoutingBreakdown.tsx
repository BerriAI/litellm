import { useState } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { DataTable } from "@/components/shared/DataTable";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { addDays, format } from "date-fns";

type RoutingRow = components["schemas"]["RoutingUsageRow"];

interface RoutingBreakdownProps {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  teamIds?: string[];
  userId?: string;
  initialModel?: string;
  enabled: boolean;
}

const money = (value: number) =>
  `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 8 })}`;
const origin = (row: RoutingRow) => {
  if (row.attribution === "router") return row.router_name;
  return row.attribution === "direct" ? "Direct" : "Unattributed";
};

export default function RoutingBreakdown({
  accessToken,
  startTime,
  endTime,
  teamIds,
  userId,
  initialModel,
  enabled,
}: RoutingBreakdownProps) {
  const [direction, setDirection] = useState(initialModel ? "model" : "router");
  const [selectedRouter, setSelectedRouter] = useState("");
  const [selectedModel, setSelectedModel] = useState(initialModel ?? "");
  const hasWindow = !!startTime && !!endTime;
  const { data, isLoading, error, refetch } = $api.useQuery(
    "get",
    "/spend/routing",
    {
      params: {
        query: {
          start_date: startTime ? format(startTime, "yyyy-MM-dd") : "",
          end_date: endTime ? format(addDays(endTime, 1), "yyyy-MM-dd") : "",
          team_ids: teamIds,
          user_id: userId,
        },
      },
    },
    { enabled: enabled && !!accessToken && hasWindow },
  );
  const allRows = data?.results ?? [];
  const routers = [...new Set(allRows.flatMap((row) => (row.router_name ? [row.router_name] : [])))].sort();
  const models = [...new Set(allRows.map((row) => row.model))].sort();
  const rows = allRows.filter((row) =>
    direction === "router"
      ? row.attribution === "router" && (!selectedRouter || row.router_name === selectedRouter)
      : !selectedModel || row.model === selectedModel,
  );
  const requests = rows.reduce((sum, row) => sum + row.requests, 0);
  const routedRequests = rows.reduce((sum, row) => sum + (row.attribution === "router" ? row.requests : 0), 0);
  const knownCosts = rows.reduce((sum, row) => sum + row.classifier_cost_known_requests, 0);
  const columns: ColumnDef<RoutingRow>[] = [
    { header: "Origin", id: "origin", accessorFn: origin },
    { header: "Destination model", accessorKey: "model" },
    { header: "Provider", accessorKey: "provider" },
    { header: "Requests", accessorKey: "requests", meta: { numeric: true } },
    {
      header: "Traffic share",
      id: "share",
      meta: { numeric: true },
      cell: ({ row }) => `${requests ? ((100 * row.original.requests) / requests).toFixed(1) : "0.0"}%`,
    },
    { header: "Cache hits", accessorKey: "cache_hits", meta: { numeric: true } },
    { header: "Failed attempts", accessorKey: "failed_attempts", meta: { numeric: true } },
    {
      header: "Inference spend",
      accessorKey: "inference_spend",
      meta: { numeric: true },
      cell: ({ row }) => money(row.original.inference_spend),
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Routing breakdown</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Historical destinations and origins from retained spend logs. Requests count successful user inference,
          including cache hits. Failed attempts and internal calls do not increase traffic share.
        </p>
        <Tabs value={direction} onValueChange={setDirection}>
          <TabsList aria-label="Routing direction">
            <TabsTrigger value="router">Router destinations</TabsTrigger>
            <TabsTrigger value="model">Model origins</TabsTrigger>
          </TabsList>
        </Tabs>
        {direction === "router" ? (
          <label className="flex items-center gap-3 text-sm">
            Router
            <select
              aria-label="Router"
              className="rounded-md border bg-background p-2"
              value={selectedRouter}
              onChange={(event) => setSelectedRouter(event.target.value)}
            >
              <option value="">All recorded routers</option>
              {routers.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <label className="flex items-center gap-3 text-sm">
            Model
            <select
              aria-label="Model"
              className="rounded-md border bg-background p-2"
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            >
              <option value="">All recorded models</option>
              {models.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        )}
        {error ? (
          <div role="alert">
            Could not load routing breakdown.
            <Button variant="outline" onClick={() => refetch()}>
              Retry
            </Button>
          </div>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <p className="text-sm text-muted-foreground">Successful requests</p>
                <p className="text-xl font-semibold">{requests.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Inference spend</p>
                <p className="text-xl font-semibold">
                  {money(rows.reduce((sum, row) => sum + row.inference_spend, 0))}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Recorded classifier overhead</p>
                <p className="text-xl font-semibold">
                  {money(rows.reduce((sum, row) => sum + row.classifier_spend, 0))}
                </p>
              </div>
            </div>
            <DataTable
              columns={columns}
              data={rows}
              isLoading={isLoading}
              size="compact"
              noDataMessage="No retained routing records for this selection"
            />
            <p className="text-sm text-muted-foreground">
              Classifier cost is known for {knownCosts} of {routedRequests} routed successful requests. Overhead is
              separate from inference spend and counted once per recorded call. Missing and orphan classifier costs are
              unknown. Billing totals are unchanged.
            </p>
          </>
        )}
        <p className="text-sm text-muted-foreground">
          UTC calendar dates. This view covers retained logs only
          {data?.configured_retention ? ` (configured retention: ${data.configured_retention})` : ""}. Older records
          without reliable origin are Unattributed. Deleted routers remain visible while their logs are retained.
        </p>
        {data?.spend_logs_disabled && (
          <p role="alert" className="text-sm text-amber-600">
            Spend logging is disabled. New requests will not appear in this view.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
