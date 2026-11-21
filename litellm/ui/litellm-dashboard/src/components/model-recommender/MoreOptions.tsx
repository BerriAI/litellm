"use client";

import { useTranslation } from "react-i18next";
import { Panel } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { Logo } from "@/components/molecules/logo/Logo";
import { Badge } from "@/components/ui/badge";
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";
import { compactNumber, metricValue, type Recommendation } from "./api";
import { ModelActions } from "./ModelActions";

function price(value: number | null | undefined) {
  return value == null ? "Unknown" : `$${(value * 1_000_000).toFixed(2)}`;
}

export function MoreOptions({ models }: { models: readonly Recommendation[] }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  if (!models.length) return null;
  return (
    <Panel title={t("moreOptions")}>
      <Table aria-label={t("moreOptions")}>
        <TableHeader>
          <TableRow>
            <TableHead>#</TableHead>
            <TableHead>{t("modelLabel")}</TableHead>
            <TableHead>{t("priorities.intelligence")}</TableHead>
            <TableHead>{t("latency")}</TableHead>
            <TableHead>{t("contextLabel")}</TableHead>
            <TableHead>{t("inputPrice")}</TableHead>
            <TableHead>{t("outputPrice")}</TableHead>
            <TableHead>{t("actions")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model, index) => (
            <TableRow key={model.model_group}>
              <TableCell>{index + 4}</TableCell>
              <TableCell>
                <span className="flex items-center gap-2">
                  <Logo provider={model.providers[0] ?? ""} className="size-5 shrink-0" />
                  {model.model_group}
                </span>
                {model.on_gateway === false && (
                  <Badge variant="secondary" className="mt-1">
                    {t("notOnGateway")}
                  </Badge>
                )}
              </TableCell>
              <TableCell>
                {model.signals.intelligence.raw == null
                  ? t("noData")
                  : `${Math.round(model.signals.intelligence.raw * 100)}%`}
              </TableCell>
              <TableCell>{metricValue("speed", model.latency_seconds)}</TableCell>
              <TableCell>{model.context_window == null ? t("noData") : compactNumber(model.context_window)}</TableCell>
              <TableCell>{price(model.input_cost_per_token)}</TableCell>
              <TableCell>{price(model.output_cost_per_token)}</TableCell>
              <TableCell className="min-w-72">
                <ModelActions model={model} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Panel>
  );
}
