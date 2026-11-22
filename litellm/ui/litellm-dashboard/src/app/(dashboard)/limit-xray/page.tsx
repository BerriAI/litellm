"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { ScanLine } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface LimitLayer {
  scope: string;
  name: string;
  kind: "budget" | "rpm";
  limit: number;
  used: number;
  remaining: number;
  requests_left: number | null;
  resets_at: string | null;
  change_with: string;
}
interface LimitXray {
  key_alias: string;
  key_aliases: string[];
  avg_request_usd: number;
  layers: LimitLayer[];
  unlimited: string[];
}

const amount = (layer: LimitLayer, value: number) =>
  layer.kind === "budget" ? `$${value.toFixed(layer.limit < 1 ? 4 : 2)}` : `${value.toFixed(0)} rpm`;

export default function LimitXrayPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [alias, setAlias] = useState("");
  const [data, setData] = useState<LimitXray | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient
      .get<LimitXray>(`/what_if/limit_xray?key_alias=${encodeURIComponent(alias)}`, { accessToken })
      .then(setData);
  }, [accessToken, alias]);

  const binding = data?.layers[0];

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <ScanLine />
          {t("whatIf.limitXray.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.limitXray.description")}</PageHeaderDescription>
      </PageHeader>
      <div className="flex flex-wrap gap-2">
        {data?.key_aliases.map((key) => (
          <Button
            key={key}
            size="sm"
            variant={key === data.key_alias ? "default" : "outline"}
            onClick={() => setAlias(key)}
          >
            {key}
          </Button>
        ))}
      </div>
      {binding && (
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">{t("whatIf.limitXray.bindsFirst")}</CardTitle>
          </CardHeader>
          <CardContent className="text-xl font-semibold">
            {t("whatIf.limitXray.verdict", {
              scope: binding.scope,
              name: binding.name,
              kind: binding.kind,
              count: binding.requests_left ?? 0,
            })}
            <div className="mt-1 font-mono text-sm font-normal text-muted-foreground">{binding.change_with}</div>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>{t("whatIf.limitXray.layers", { alias: data?.key_alias })}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {data?.layers.map((layer, index) => {
            const percent = Math.min((layer.used / Math.max(layer.limit, 0.000001)) * 100, 100);
            return (
              <div key={`${layer.scope}-${layer.kind}`} className="grid grid-cols-[220px_1fr_260px] items-center gap-4">
                <div>
                  <div className="font-medium">
                    {layer.scope} {layer.kind}
                  </div>
                  <div className="text-xs text-muted-foreground">{layer.name}</div>
                </div>
                <div className="h-3 overflow-hidden rounded bg-muted">
                  <div
                    className={index === 0 ? "h-3 bg-destructive" : "h-3 bg-primary"}
                    style={{ width: `${percent}%` }}
                  />
                </div>
                <div className="flex items-center justify-between gap-2 text-sm">
                  <span>
                    {amount(layer, layer.used)} / {amount(layer, layer.limit)}
                  </span>
                  <Badge variant={index === 0 ? "destructive" : "secondary"}>
                    {t("whatIf.limitXray.requestsLeft", { count: layer.requests_left ?? 0 })}
                  </Badge>
                </div>
              </div>
            );
          })}
          {data && data.unlimited.length > 0 && (
            <div className="text-sm text-muted-foreground">
              {t("whatIf.limitXray.unlimited", { scopes: data.unlimited.join(", ") })}
            </div>
          )}
        </CardContent>
      </Card>
    </Page>
  );
}
