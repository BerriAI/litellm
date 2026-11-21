"use client";

import React, { useEffect, useState } from "react";
import { SearchX } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface CacheTurn {
  request_id: string;
  start_time: string;
  key_alias: string;
  session_id: string;
  model: string;
  prompt_tokens: number;
  cache_read_tokens: number;
  missed_tokens: number;
  wasted_usd: number;
  verdict: "hit" | "client" | "gateway" | "provider" | "first";
  reason: string;
  changed_text: string | null;
}
interface CacheMissResponse {
  turns: CacheTurn[];
  wasted_usd_by_cause: Record<string, number>;
  wasted_usd_per_day: number;
}

const CAUSES: Record<string, string> = {
  client: "Client changed the prompt",
  gateway: "Gateway moved the session",
  provider: "Provider dropped the cache",
};
const usd = (value: number) => `$${value.toFixed(value < 0.01 ? 4 : 2)}`;

export default function CacheMissPage() {
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<CacheMissResponse | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<CacheMissResponse>("/what_if/cache_miss", { accessToken }).then(setData);
  }, [accessToken]);

  const sessions = Array.from(new Set(data?.turns.map((turn) => turn.session_id) ?? []));

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <SearchX />
          Cache-Miss Explainer
        </PageHeaderTitle>
        <PageHeaderDescription>
          Why each turn did or did not read the prompt cache, and who has to fix it: the client, the gateway or the
          provider.
        </PageHeaderDescription>
      </PageHeader>
      <div className="grid grid-cols-3 gap-4">
        {Object.entries(CAUSES).map(([cause, label]) => (
          <Card key={cause}>
            <CardHeader>
              <CardTitle className="text-sm text-muted-foreground">{label}</CardTitle>
            </CardHeader>
            <CardContent className="text-2xl font-semibold">
              {data ? usd(data.wasted_usd_by_cause[cause] ?? 0) : "-"}
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                re-billed · {data?.turns.filter((turn) => turn.verdict === cause).length ?? 0} turns
              </span>
            </CardContent>
          </Card>
        ))}
      </div>
      {sessions.map((session) => (
        <Card key={session}>
          <CardHeader>
            <CardTitle>
              {data?.turns.find((turn) => turn.session_id === session)?.key_alias} · session {session}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Turn</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Prompt tokens</TableHead>
                  <TableHead>Read from cache</TableHead>
                  <TableHead>Verdict</TableHead>
                  <TableHead>Why</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data?.turns
                  .filter((turn) => turn.session_id === session)
                  .map((turn, index) => (
                    <React.Fragment key={turn.request_id}>
                      <TableRow
                        className="cursor-pointer"
                        onClick={() => setOpen(open === turn.request_id ? null : turn.request_id)}
                      >
                        <TableCell>{index + 1}</TableCell>
                        <TableCell>{turn.model}</TableCell>
                        <TableCell>{turn.prompt_tokens.toLocaleString()}</TableCell>
                        <TableCell>{turn.cache_read_tokens.toLocaleString()}</TableCell>
                        <TableCell>
                          {turn.verdict === "hit" || turn.verdict === "first" ? (
                            <Badge variant="secondary">{turn.verdict}</Badge>
                          ) : (
                            <Badge variant="destructive">miss · {turn.verdict}</Badge>
                          )}
                        </TableCell>
                        <TableCell className="whitespace-normal">{turn.reason}</TableCell>
                      </TableRow>
                      {open === turn.request_id && turn.changed_text && (
                        <TableRow>
                          <TableCell colSpan={6} className="whitespace-normal bg-muted font-mono text-xs">
                            {turn.changed_text}
                          </TableCell>
                        </TableRow>
                      )}
                    </React.Fragment>
                  ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ))}
    </Page>
  );
}
