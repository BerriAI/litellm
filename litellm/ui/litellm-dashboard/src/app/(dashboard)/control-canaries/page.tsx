"use client";

import { useTranslation } from "react-i18next";
import React, { useState } from "react";
import { ShieldCheck } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface CanaryCell {
  route: string;
  status_code: number;
  enforced: boolean;
  detail: string;
}
interface CanaryResult {
  key_alias: string;
  control: string;
  configured: string;
  probe: string;
  cells: CanaryCell[];
}
interface CanaryResponse {
  ran_at: string;
  results: CanaryResult[];
}

const ROUTES = ["/v1/chat/completions", "/v1/messages", "/v1/responses"];

export default function ControlCanariesPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [guardrail, setGuardrail] = useState("no-codenames");
  const [probe, setProbe] = useState("Summarize the launch plan for project nightfall");
  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<CanaryResponse | null>(null);

  const run = async () => {
    setRunning(true);
    try {
      setResponse(
        await apiClient.post<CanaryResponse>("/what_if/canaries/run", {
          accessToken,
          body: { guardrail_probes: { [guardrail]: probe } },
        }),
      );
    } finally {
      setRunning(false);
    }
  };

  const cells = response?.results.flatMap((result) => result.cells) ?? [];
  const leaks = cells.filter((cell) => !cell.enforced).length;

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <ShieldCheck />
          {t("whatIf.canaries.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.canaries.description")}</PageHeaderDescription>
        <PageHeaderControls>
          <Button onClick={run} disabled={running}>
            {running ? t("whatIf.canaries.probing") : t("whatIf.canaries.run")}
          </Button>
        </PageHeaderControls>
      </PageHeader>
      <Card>
        <CardHeader>
          <CardTitle>{t("whatIf.canaries.guardrailProbe")}</CardTitle>
        </CardHeader>
        <CardContent className="flex gap-3">
          <Input className="w-56" value={guardrail} onChange={(event) => setGuardrail(event.target.value)} />
          <Input value={probe} onChange={(event) => setProbe(event.target.value)} />
        </CardContent>
      </Card>
      {response && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              {t("whatIf.canaries.held", { held: cells.length - leaks, total: cells.length })}
              {leaks > 0 ? (
                <Badge variant="destructive">{t("whatIf.canaries.notEnforced", { count: leaks })}</Badge>
              ) : (
                <Badge variant="secondary">{t("whatIf.canaries.allEnforced")}</Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("whatIf.canaries.key")}</TableHead>
                  <TableHead>{t("whatIf.canaries.configured")}</TableHead>
                  <TableHead>{t("whatIf.canaries.probe")}</TableHead>
                  {ROUTES.map((route) => (
                    <TableHead key={route}>{route}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {response.results.map((result) => (
                  <TableRow key={`${result.key_alias}-${result.control}-${result.configured}`}>
                    <TableCell className="font-medium">{result.key_alias}</TableCell>
                    <TableCell>{result.configured}</TableCell>
                    <TableCell className="text-muted-foreground">{result.probe}</TableCell>
                    {ROUTES.map((route) => {
                      const cell = result.cells.find((candidate) => candidate.route === route);
                      if (!cell) return <TableCell key={route}>-</TableCell>;
                      return (
                        <TableCell key={route} title={cell.detail}>
                          {cell.enforced ? (
                            <Badge variant="secondary">
                              {t("whatIf.canaries.cellHeld", { status: cell.status_code })}
                            </Badge>
                          ) : (
                            <Badge variant="destructive">
                              {t("whatIf.canaries.cellLeaked", { status: cell.status_code })}
                            </Badge>
                          )}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </Page>
  );
}
