"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { ArrowLeftRight } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface SwapRow {
  team: string;
  model: string;
  successor: string;
  retires: string | null;
  now: number;
  after: number;
  delta: number;
}
interface SwapTeam {
  team: string;
  budget: number | null;
  now: number;
  after: number;
  budget_lasts_days_now: number | null;
  budget_lasts_days_after: number | null;
}
interface SwapBill {
  window_days: number;
  rows: SwapRow[];
  teams: SwapTeam[];
  now: number;
  after: number;
}

const usd = (value: number) => `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const signed = (value: number) => `${value >= 0 ? "+" : "-"}${usd(Math.abs(value))}`;
const lasts = (days: number | null, window: number) =>
  days === null ? "-" : days >= window ? `${window}+` : days.toFixed(0);

export default function SwapBillPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<SwapBill | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<SwapBill>("/what_if/swap_bill", { accessToken }).then(setData);
  }, [accessToken]);

  const stats = [
    ["whatIf.swapBill.now", data ? usd(data.now) : "-"],
    ["whatIf.swapBill.after", data ? usd(data.after) : "-"],
    ["whatIf.swapBill.delta", data ? signed(data.after - data.now) : "-"],
  ] as const;

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <ArrowLeftRight />
          {t("whatIf.swapBill.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.swapBill.description")}</PageHeaderDescription>
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
                <TableHead>{t("whatIf.swapBill.team")}</TableHead>
                <TableHead>{t("whatIf.swapBill.retiring")}</TableHead>
                <TableHead>{t("whatIf.swapBill.successor")}</TableHead>
                <TableHead>{t("whatIf.swapBill.retires")}</TableHead>
                <TableHead>{t("whatIf.swapBill.paidNow")}</TableHead>
                <TableHead>{t("whatIf.swapBill.onSuccessor")}</TableHead>
                <TableHead>{t("whatIf.swapBill.change")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.rows.map((row) => (
                <TableRow key={`${row.team}-${row.model}`}>
                  <TableCell className="font-medium">{row.team}</TableCell>
                  <TableCell>{row.model}</TableCell>
                  <TableCell>{row.successor}</TableCell>
                  <TableCell>{row.retires ?? "-"}</TableCell>
                  <TableCell>{usd(row.now)}</TableCell>
                  <TableCell>{usd(row.after)}</TableCell>
                  <TableCell>
                    <Badge variant={row.delta > 0 ? "destructive" : "secondary"}>
                      {signed(row.delta)} ({(row.after / row.now).toFixed(1)}x)
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>{t("whatIf.swapBill.budgets")}</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("whatIf.swapBill.team")}</TableHead>
                <TableHead>{t("whatIf.swapBill.budget")}</TableHead>
                <TableHead>{t("whatIf.swapBill.spendNow")}</TableHead>
                <TableHead>{t("whatIf.swapBill.spendAfter")}</TableHead>
                <TableHead>{t("whatIf.swapBill.lastsNow")}</TableHead>
                <TableHead>{t("whatIf.swapBill.lastsAfter")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.teams.map((team) => (
                <TableRow key={team.team}>
                  <TableCell className="font-medium">{team.team}</TableCell>
                  <TableCell>{team.budget === null ? "-" : usd(team.budget)}</TableCell>
                  <TableCell>{usd(team.now)}</TableCell>
                  <TableCell>{usd(team.after)}</TableCell>
                  <TableCell>{lasts(team.budget_lasts_days_now, data.window_days)}</TableCell>
                  <TableCell className="font-semibold">
                    {lasts(team.budget_lasts_days_after, data.window_days)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Page>
  );
}
