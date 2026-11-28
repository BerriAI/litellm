"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { History } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface RecallRow {
  team: string;
  model: string;
  first_day: string;
  last_day: string;
  days: number;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  logged: number;
  corrected: number;
  delta: number;
}
interface BillRecall {
  rows: RecallRow[];
  logged: number;
  corrected: number;
  delta: number;
  days_checked: number;
}

const usd = (value: number) =>
  `$${value.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}`;
const signed = (value: number) => `${value >= 0 ? "+" : "-"}${usd(Math.abs(value))}`;

export default function BillRecallPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<BillRecall | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<BillRecall>("/what_if/bill_recall", { accessToken }).then(setData);
  }, [accessToken]);

  const stats = [
    ["whatIf.billRecall.logged", data ? usd(data.logged) : "-"],
    ["whatIf.billRecall.corrected", data ? usd(data.corrected) : "-"],
    ["whatIf.billRecall.delta", data ? signed(data.delta) : "-"],
  ] as const;

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <History />
          {t("whatIf.billRecall.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.billRecall.description")}</PageHeaderDescription>
      </PageHeader>
      <div className="grid grid-cols-3 gap-4">
        {stats.map(([label, value]) => (
          <Card key={label}>
            <CardHeader>
              <CardTitle className="text-sm text-muted-foreground">{t(label)}</CardTitle>
            </CardHeader>
            <CardContent className="text-2xl font-semibold">{value}</CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("whatIf.billRecall.team")}</TableHead>
                <TableHead>{t("whatIf.billRecall.model")}</TableHead>
                <TableHead>{t("whatIf.billRecall.days")}</TableHead>
                <TableHead>{t("whatIf.billRecall.tokens")}</TableHead>
                <TableHead>{t("whatIf.billRecall.loggedAt")}</TableHead>
                <TableHead>{t("whatIf.billRecall.correctedTo")}</TableHead>
                <TableHead>{t("whatIf.billRecall.owed")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.rows.map((row) => (
                <TableRow key={`${row.team}-${row.model}`}>
                  <TableCell className="font-medium">{row.team}</TableCell>
                  <TableCell>{row.model}</TableCell>
                  <TableCell>
                    {row.first_day} - {row.last_day} <Badge variant="secondary">{row.days}d</Badge>
                  </TableCell>
                  <TableCell>{(row.prompt_tokens + row.completion_tokens).toLocaleString()}</TableCell>
                  <TableCell>{usd(row.logged)}</TableCell>
                  <TableCell>{usd(row.corrected)}</TableCell>
                  <TableCell className="font-semibold">{signed(row.delta)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Page>
  );
}
