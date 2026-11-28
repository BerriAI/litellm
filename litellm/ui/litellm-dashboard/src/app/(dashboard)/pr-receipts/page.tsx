"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { ReceiptText } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface PrReceipt {
  url: string;
  repo: string;
  number: number;
  spend: number;
  total_tokens: number;
  requests: number;
  tool_calls: number;
  agent_minutes: number;
  key_aliases: string[];
  models: string[];
  runs: string[];
}
interface PrReceipts {
  receipts: PrReceipt[];
  spend_with_pr: number;
  spend_without_pr: number;
}

const usd = (value: number) => `$${value.toFixed(4)}`;

export default function PrReceiptsPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<PrReceipts | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<PrReceipts>("/what_if/pr_receipts", { accessToken }).then(setData);
  }, [accessToken]);

  const stats = [
    ["whatIf.prReceipts.shipped", data ? usd(data.spend_with_pr) : "-"],
    ["whatIf.prReceipts.noPr", data ? usd(data.spend_without_pr) : "-"],
    ["whatIf.prReceipts.count", data?.receipts.length ?? "-"],
  ] as const;

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <ReceiptText />
          {t("whatIf.prReceipts.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.prReceipts.description")}</PageHeaderDescription>
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
                <TableHead>{t("whatIf.prReceipts.pr")}</TableHead>
                <TableHead>{t("whatIf.prReceipts.who")}</TableHead>
                <TableHead>{t("whatIf.prReceipts.models")}</TableHead>
                <TableHead>{t("whatIf.prReceipts.requests")}</TableHead>
                <TableHead>{t("whatIf.prReceipts.toolCalls")}</TableHead>
                <TableHead>{t("whatIf.prReceipts.tokens")}</TableHead>
                <TableHead>{t("whatIf.prReceipts.cost")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.receipts.map((receipt) => (
                <TableRow key={receipt.url}>
                  <TableCell>
                    <a className="font-medium underline" href={receipt.url} target="_blank" rel="noreferrer">
                      {receipt.repo} #{receipt.number}
                    </a>
                  </TableCell>
                  <TableCell>
                    {receipt.key_aliases.map((alias) => (
                      <Badge key={alias} variant="secondary">
                        {alias}
                      </Badge>
                    ))}
                  </TableCell>
                  <TableCell>{receipt.models.join(", ")}</TableCell>
                  <TableCell>{receipt.requests}</TableCell>
                  <TableCell>{receipt.tool_calls}</TableCell>
                  <TableCell>{receipt.total_tokens.toLocaleString()}</TableCell>
                  <TableCell className="font-semibold">{usd(receipt.spend)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Page>
  );
}
