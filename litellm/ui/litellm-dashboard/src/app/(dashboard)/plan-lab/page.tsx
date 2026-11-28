"use client";

import { useTranslation } from "react-i18next";
import React, { useState } from "react";
import { FlaskConical } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface PlanCustomer {
  customer: string;
  requests: number;
  cost_today: number;
  cost_under_plan: number;
  margin: number;
  blocked_requests: number;
  rerouted_requests: number;
}
interface PlanReplay {
  customers: PlanCustomer[];
  requests_replayed: number;
  revenue: number;
  cost_under_plan: number;
  margin: number;
  margin_pct: number;
  customers_losing_money: number;
  blocked_requests: number;
  rerouted_requests: number;
}

const usd = (value: number) => `${value < 0 ? "-" : ""}$${Math.abs(value).toFixed(2)}`;
const MODES = ["allow", "block", "fallback"] as const;

export default function PlanLabPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [price, setPrice] = useState("20");
  const [included, setIncluded] = useState("15");
  const [mode, setMode] = useState<(typeof MODES)[number]>("allow");
  const [fallback, setFallback] = useState("gpt-5.4-nano");
  const [replay, setReplay] = useState<PlanReplay | null>(null);

  const run = async () =>
    setReplay(
      await apiClient.post<PlanReplay>("/what_if/plan_lab/replay", {
        accessToken,
        body: {
          monthly_price: Number(price),
          included_spend: Number(included),
          over_budget: mode,
          fallback_model: fallback,
        },
      }),
    );

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <FlaskConical />
          {t("whatIf.planLab.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.planLab.description")}</PageHeaderDescription>
        <PageHeaderControls>
          <Button onClick={run}>{t("whatIf.planLab.replay")}</Button>
        </PageHeaderControls>
      </PageHeader>
      <Card>
        <CardHeader>
          <CardTitle>{t("whatIf.planLab.draftPlan")}</CardTitle>
        </CardHeader>
        <CardContent className="flex items-end gap-4">
          <div>
            <Label>{t("whatIf.planLab.price")}</Label>
            <Input className="w-40" value={price} onChange={(event) => setPrice(event.target.value)} />
          </div>
          <div>
            <Label>{t("whatIf.planLab.included")}</Label>
            <Input className="w-40" value={included} onChange={(event) => setIncluded(event.target.value)} />
          </div>
          <div>
            <Label>{t("whatIf.planLab.overBudget")}</Label>
            <div className="flex gap-1">
              {MODES.map((candidate) => (
                <Button
                  key={candidate}
                  variant={mode === candidate ? "default" : "outline"}
                  onClick={() => setMode(candidate)}
                >
                  {t(`whatIf.planLab.${candidate}`)}
                </Button>
              ))}
            </div>
          </div>
          {mode === "fallback" && (
            <Input className="w-48" value={fallback} onChange={(event) => setFallback(event.target.value)} />
          )}
        </CardContent>
      </Card>
      {replay && (
        <>
          <div className="grid grid-cols-4 gap-4">
            {[
              [t("whatIf.planLab.revenue"), usd(replay.revenue)],
              [t("whatIf.planLab.modelCost"), usd(replay.cost_under_plan)],
              [t("whatIf.planLab.margin"), `${usd(replay.margin)} (${replay.margin_pct.toFixed(0)}%)`],
              [
                t("whatIf.planLab.losing"),
                t("whatIf.planLab.losingValue", {
                  count: replay.customers_losing_money,
                  total: replay.customers.length,
                }),
              ],
            ].map(([label, value]) => (
              <Card key={label}>
                <CardHeader>
                  <CardTitle className="text-sm text-muted-foreground">{label}</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-semibold">{value}</CardContent>
              </Card>
            ))}
          </div>
          <Card>
            <CardHeader>
              <CardTitle>
                {t("whatIf.planLab.summary", {
                  requests: replay.requests_replayed.toLocaleString(),
                  blocked: replay.blocked_requests.toLocaleString(),
                  rerouted: replay.rerouted_requests.toLocaleString(),
                })}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("whatIf.planLab.customer")}</TableHead>
                    <TableHead>{t("whatIf.planLab.requests")}</TableHead>
                    <TableHead>{t("whatIf.planLab.costToday")}</TableHead>
                    <TableHead>{t("whatIf.planLab.costUnderPlan")}</TableHead>
                    <TableHead>{t("whatIf.planLab.blocked")}</TableHead>
                    <TableHead>{t("whatIf.planLab.rerouted")}</TableHead>
                    <TableHead>{t("whatIf.planLab.margin")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {replay.customers.slice(0, 12).map((customer) => (
                    <TableRow key={customer.customer}>
                      <TableCell className="font-medium">{customer.customer}</TableCell>
                      <TableCell>{customer.requests.toLocaleString()}</TableCell>
                      <TableCell>{usd(customer.cost_today)}</TableCell>
                      <TableCell>{usd(customer.cost_under_plan)}</TableCell>
                      <TableCell>{customer.blocked_requests.toLocaleString()}</TableCell>
                      <TableCell>{customer.rerouted_requests.toLocaleString()}</TableCell>
                      <TableCell>
                        <Badge variant={customer.margin < 0 ? "destructive" : "secondary"}>
                          {usd(customer.margin)}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </Page>
  );
}
