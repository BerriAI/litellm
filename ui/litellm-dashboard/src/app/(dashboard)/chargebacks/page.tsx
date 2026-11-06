"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { dailyActivityAggregatedCall } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatUSD, monthRange } from "../customers/_components/customerPortalData";

type GroupBy = "customer" | "key" | "model";
const DIMENSION_LABELS: Record<GroupBy, string | null> = { customer: null, key: "Key", model: "Model" };
const GROUP_LABELS: Record<GroupBy, string> = { customer: "Group by customer", key: "Customer and key", model: "Customer and model" };

interface Row {
  customer: string;
  dimension: string | null;
  requests: number;
  inputTokens: number;
  outputTokens: number;
  spend: number;
}

interface Metrics { spend?: number; api_requests?: number; prompt_tokens?: number; completion_tokens?: number }
interface Metric { metrics?: Metrics; metadata?: { key_alias?: string | null } }

const add = (row: Row, m: Metrics | undefined) => {
  row.requests += m?.api_requests ?? 0;
  row.inputTokens += m?.prompt_tokens ?? 0;
  row.outputTokens += m?.completion_tokens ?? 0;
  row.spend += m?.spend ?? 0;
};

const monthLabel = (back: number) => {
  const { start } = monthRange(back);
  return start.toLocaleString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
};

export default function ChargebacksPage() {
  const { accessToken } = useAuthorized();
  const [monthsBack, setMonthsBack] = useState(0);
  const [groupBy, setGroupBy] = useState<GroupBy>("customer");
  const customersQuery = useCustomers(100);
  const customers = useMemo(() => customersQuery.data?.pages.flat() ?? [], [customersQuery.data]);
  const aliasOf = useMemo(() => new Map(customers.map((c) => [c.user_id, c.alias || c.user_id])), [customers]);

  const report = useQuery({
    queryKey: ["customer-portal", "chargebacks", monthsBack, groupBy, customers.map((c) => c.user_id).join(",")],
    enabled: Boolean(accessToken) && customersQuery.isSuccess,
    queryFn: async (): Promise<Row[]> => {
      const { start, end } = monthRange(monthsBack);
      const rows = new Map<string, Row>();
      const upsert = (customer: string, dimension: string | null, m: Metrics | undefined) => {
        const id = `${customer}\u0000${dimension ?? ""}`;
        const row = rows.get(id) ?? { customer, dimension, requests: 0, inputTokens: 0, outputTokens: 0, spend: 0 };
        add(row, m);
        rows.set(id, row);
      };
      if (groupBy === "customer") {
        const res = await dailyActivityAggregatedCall("customer", { accessToken: accessToken!, startTime: start, endTime: end });
        for (const day of res.results) {
          for (const [id, e] of Object.entries((day.breakdown?.entities ?? {}) as Record<string, Metric>)) upsert(id, null, e.metrics);
        }
      } else {
        await Promise.all(
          customers.map(async (c) => {
            const res = await dailyActivityAggregatedCall("customer", { accessToken: accessToken!, startTime: start, endTime: end, entityIds: [c.user_id] });
            for (const day of res.results) {
              const bucket = (groupBy === "key" ? day.breakdown?.api_keys : day.breakdown?.models) ?? {};
              for (const [id, m] of Object.entries(bucket as Record<string, Metric>)) {
                upsert(c.user_id, groupBy === "key" ? m.metadata?.key_alias || `${id.slice(0, 12)}...` : id, m.metrics);
              }
            }
          }),
        );
      }
      return [...rows.values()].filter((r) => r.requests > 0 || r.spend > 0).sort((a, b) => b.spend - a.spend);
    },
  });

  const rows = report.data ?? [];
  const total = rows.reduce((acc, r) => { add(acc, { spend: r.spend, api_requests: r.requests, prompt_tokens: r.inputTokens, completion_tokens: r.outputTokens }); return acc; }, { customer: "", dimension: null, requests: 0, inputTokens: 0, outputTokens: 0, spend: 0 } as Row);
  const dimensionLabel = DIMENSION_LABELS[groupBy];

  const exportCsv = () => {
    const header = ["customer_id", "customer", ...(dimensionLabel ? [dimensionLabel.toLowerCase()] : []), "requests", "input_tokens", "output_tokens", "spend_usd"];
    const lines = rows.map((r) => [r.customer, aliasOf.get(r.customer) ?? r.customer, ...(dimensionLabel ? [r.dimension ?? ""] : []), r.requests, r.inputTokens, r.outputTokens, r.spend.toFixed(6)]);
    const csv = [header, ...lines].map((l) => l.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `chargebacks-${monthLabel(monthsBack).replace(" ", "-").toLowerCase()}-${groupBy}.csv` });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full space-y-4 px-6 py-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">Chargebacks</h1>
          <p className="truncate text-sm text-muted-foreground">Spend per customer for a month, from spend logs.</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={String(monthsBack)} onValueChange={(v) => setMonthsBack(Number(v))}>
            <SelectTrigger className="w-44"><SelectValue>{monthLabel(monthsBack)}</SelectValue></SelectTrigger>
            <SelectContent>{[0, 1, 2].map((b) => <SelectItem key={b} value={String(b)}>{monthLabel(b)}</SelectItem>)}</SelectContent>
          </Select>
          <Select value={groupBy} onValueChange={(v) => setGroupBy(v as GroupBy)}>
            <SelectTrigger className="w-48"><SelectValue>{GROUP_LABELS[groupBy]}</SelectValue></SelectTrigger>
            <SelectContent>
              {(Object.keys(GROUP_LABELS) as GroupBy[]).map((g) => <SelectItem key={g} value={g}>{GROUP_LABELS[g]}</SelectItem>)}
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={exportCsv} disabled={rows.length === 0}><Download className="size-4" /> Export CSV</Button>
        </div>
      </div>
      <div className="rounded-lg border bg-card">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">Customer</TableHead>
              {dimensionLabel && <TableHead>{dimensionLabel}</TableHead>}
              <TableHead className="text-right">Requests</TableHead>
              <TableHead className="text-right">Input tokens</TableHead>
              <TableHead className="text-right">Output tokens</TableHead>
              <TableHead className="text-right pr-4">Spend</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {report.isLoading && <TableRow><TableCell colSpan={6} className="py-8 text-center text-muted-foreground">Loading...</TableCell></TableRow>}
            {!report.isLoading && rows.length === 0 && <TableRow><TableCell colSpan={6} className="py-8 text-center text-muted-foreground">No customer traffic in {monthLabel(monthsBack)}.</TableCell></TableRow>}
            {rows.map((r) => (
              <TableRow key={`${r.customer}-${r.dimension ?? ""}`}>
                <TableCell className="pl-4">
                  <p className="font-medium">{aliasOf.get(r.customer) ?? r.customer}</p>
                  <p className="font-mono text-[11px] text-muted-foreground">{r.customer}</p>
                </TableCell>
                {dimensionLabel && <TableCell>{r.dimension}</TableCell>}
                <TableCell className="text-right tabular-nums">{r.requests.toLocaleString()}</TableCell>
                <TableCell className="text-right tabular-nums">{r.inputTokens.toLocaleString()}</TableCell>
                <TableCell className="text-right tabular-nums">{r.outputTokens.toLocaleString()}</TableCell>
                <TableCell className="text-right tabular-nums pr-4">{formatUSD(r.spend)}</TableCell>
              </TableRow>
            ))}
            {rows.length > 0 && (
              <TableRow className="font-medium">
                <TableCell className="pl-4" colSpan={dimensionLabel ? 2 : 1}>Total</TableCell>
                <TableCell className="text-right tabular-nums">{total.requests.toLocaleString()}</TableCell>
                <TableCell className="text-right tabular-nums">{total.inputTokens.toLocaleString()}</TableCell>
                <TableCell className="text-right tabular-nums">{total.outputTokens.toLocaleString()}</TableCell>
                <TableCell className="text-right tabular-nums pr-4">{formatUSD(total.spend)}</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
