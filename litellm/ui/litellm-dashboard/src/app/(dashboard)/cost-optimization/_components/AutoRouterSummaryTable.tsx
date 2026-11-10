import { useTranslation } from "react-i18next";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

import { groupKey, groupLabel, pctLabel, type AutoRouterBenchmarkGroup } from "./autoRouterBenchmarks";
import { usd } from "./costOptimizationUtils";

interface AutoRouterSummaryTableProps {
  groups: readonly AutoRouterBenchmarkGroup[];
  selectedGroup: AutoRouterBenchmarkGroup | null;
}

export default function AutoRouterSummaryTable({ groups, selectedGroup }: AutoRouterSummaryTableProps) {
  const { t } = useTranslation();
  const visibleGroups = selectedGroup ? [selectedGroup] : groups;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("costOptimization.routerSummary.title")}</CardTitle>
        <CardDescription>{t("costOptimization.routerSummary.description")}</CardDescription>
      </CardHeader>
      <CardContent>
        <Table aria-label={t("costOptimization.routerSummary.title")}>
          <TableHeader>
            <TableRow>
              <TableHead>{t("costOptimization.routerSummary.columns.router")}</TableHead>
              <TableHead className="text-right whitespace-normal">
                {t("costOptimization.routerSummary.columns.tokens")}
              </TableHead>
              <TableHead className="text-right whitespace-normal">
                {t("costOptimization.routerSummary.columns.cost")}
              </TableHead>
              <TableHead className="text-right whitespace-normal">
                {t("costOptimization.routerSummary.columns.costPerMillion")}
              </TableHead>
              <TableHead className="text-right whitespace-normal">
                {t("costOptimization.routerSummary.columns.saved")}
              </TableHead>
              <TableHead className="text-right">{t("costOptimization.routerSummary.columns.savedPct")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {visibleGroups.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                  {t("costOptimization.routerSummary.empty")}
                </TableCell>
              </TableRow>
            ) : (
              visibleGroups.map((group) => (
                <TableRow key={groupKey(group)}>
                  <TableCell className="font-medium">{groupLabel(group, groups)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {group.total_tokens == null
                      ? t("costOptimization.routerSummary.unavailable")
                      : group.total_tokens.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{usd(group.spend)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {group.total_tokens != null && group.total_tokens > 0
                      ? usd((group.spend * 1_000_000) / group.total_tokens)
                      : t("costOptimization.routerSummary.unavailable")}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {group.saved_spend == null
                      ? t("costOptimization.routerSummary.unavailable")
                      : usd(group.saved_spend)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {group.saved_pct == null
                      ? t("costOptimization.routerSummary.unavailable")
                      : pctLabel(group.saved_pct)}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
        <p className="mt-3 text-xs text-muted-foreground">{t("costOptimization.routerSummary.footnote")}</p>
      </CardContent>
    </Card>
  );
}
