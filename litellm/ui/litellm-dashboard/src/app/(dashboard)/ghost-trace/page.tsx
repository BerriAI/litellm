"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { Ghost } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface GhostStep {
  request_id: string;
  model: string;
  spend: number;
  role: string;
  summary: string;
}
interface GhostRun {
  run_id: string;
  key_alias: string;
  task: string;
  models: string[];
  spend: number;
  total_tokens: number;
  tool_calls: number;
  duration_seconds: number;
  pr_urls: string[];
  steps: GhostStep[];
}
interface GhostTrace {
  loose_requests: number;
  client_session_ids: number;
  runs: GhostRun[];
}

export default function GhostTracePage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<GhostTrace | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<GhostTrace>("/what_if/ghost_trace", { accessToken }).then(setData);
  }, [accessToken]);

  const stats = [
    ["whatIf.ghostTrace.requests", data?.loose_requests],
    ["whatIf.ghostTrace.clientSessions", data?.client_session_ids],
    ["whatIf.ghostTrace.stitched", data?.runs.length],
  ] as const;

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <Ghost />
          {t("whatIf.ghostTrace.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.ghostTrace.description")}</PageHeaderDescription>
      </PageHeader>
      <div className="grid grid-cols-3 gap-4">
        {stats.map(([label, value]) => (
          <Card key={label}>
            <CardHeader>
              <CardTitle className="text-sm text-muted-foreground">{t(label)}</CardTitle>
            </CardHeader>
            <CardContent className="text-2xl font-semibold">{value ?? "-"}</CardContent>
          </Card>
        ))}
      </div>
      {data?.runs.map((run) => (
        <Card
          key={run.run_id}
          className="cursor-pointer"
          onClick={() => setOpen(open === run.run_id ? null : run.run_id)}
        >
          <CardHeader>
            <CardTitle className="flex items-center justify-between gap-4">
              <span className="truncate">{run.task}</span>
              <span className="flex shrink-0 gap-2">
                <Badge variant="secondary">{run.key_alias}</Badge>
                <Badge variant="secondary">{t("whatIf.ghostTrace.steps", { count: run.steps.length })}</Badge>
                <Badge variant="secondary">{t("whatIf.ghostTrace.tools", { count: run.tool_calls })}</Badge>
                <Badge>${run.spend.toFixed(4)}</Badge>
              </span>
            </CardTitle>
          </CardHeader>
          {open === run.run_id && (
            <CardContent className="flex flex-col gap-2 border-l-2 border-primary pl-4">
              {run.steps.map((step, index) => (
                <div key={step.request_id} className="grid grid-cols-[32px_150px_70px_1fr_80px] gap-2 text-sm">
                  <span className="text-muted-foreground">{index + 1}</span>
                  <span>{step.model}</span>
                  <Badge variant="outline">{step.role}</Badge>
                  <span className="truncate font-mono text-xs">{step.summary}</span>
                  <span className="text-right">${step.spend.toFixed(4)}</span>
                </div>
              ))}
            </CardContent>
          )}
        </Card>
      ))}
    </Page>
  );
}
