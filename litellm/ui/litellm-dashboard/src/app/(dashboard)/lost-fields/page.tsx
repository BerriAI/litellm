"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { Unplug } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface LostField {
  model: string;
  field: string;
  fate: "dropped" | "rewritten";
  became: string[];
  sent_value: string;
  requests: number;
  example_request_id: string;
}
interface LostFields {
  window_hours: number;
  requests_scanned: number;
  requests_with_dropped_fields: number;
  fields: LostField[];
}

export default function LostFieldsPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<LostFields | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<LostFields>("/what_if/lost_fields", { accessToken }).then(setData);
  }, [accessToken]);

  const stats = [
    ["whatIf.lostFields.scanned", data?.requests_scanned ?? "-"],
    ["whatIf.lostFields.affected", data?.requests_with_dropped_fields ?? "-"],
    ["whatIf.lostFields.dropped", data?.fields.filter((field) => field.fate === "dropped").length ?? "-"],
  ] as const;

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <Unplug />
          {t("whatIf.lostFields.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.lostFields.description")}</PageHeaderDescription>
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
                <TableHead>{t("whatIf.lostFields.model")}</TableHead>
                <TableHead>{t("whatIf.lostFields.clientSent")}</TableHead>
                <TableHead>{t("whatIf.lostFields.providerGot")}</TableHead>
                <TableHead>{t("whatIf.lostFields.requests")}</TableHead>
                <TableHead>{t("whatIf.lostFields.example")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.fields.map((field) => (
                <TableRow key={`${field.model}-${field.field}-${field.sent_value}`}>
                  <TableCell className="font-medium">{field.model}</TableCell>
                  <TableCell>
                    <code>
                      {field.field}: {field.sent_value.slice(0, 40)}
                    </code>
                  </TableCell>
                  <TableCell>
                    {field.fate === "dropped" ? (
                      <Badge variant="destructive">{t("whatIf.lostFields.neverSent")}</Badge>
                    ) : (
                      <Badge variant="secondary">
                        {t("whatIf.lostFields.sentAs", { fields: field.became.join(", ") })}
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="font-semibold">{field.requests}</TableCell>
                  <TableCell>
                    <code className="text-xs text-muted-foreground">{field.example_request_id.slice(0, 22)}</code>
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
