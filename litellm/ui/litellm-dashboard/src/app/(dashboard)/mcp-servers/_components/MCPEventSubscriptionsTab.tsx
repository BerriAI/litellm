"use client";

import React from "react";
import { createParser, useQueryState } from "nuqs";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fetchMCPEventSubscriptions } from "@/components/networking";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { argumentsSummary } from "./eventSubscriptionView";

const subscriptionKeys = createQueryKeys("mcpEventSubscriptions");
const offsetParser = createParser({
  parse(value) {
    const offset = Number(value);
    return Number.isSafeInteger(offset) && offset >= 0 ? offset : null;
  },
  serialize: String,
}).withDefault(0);

const columns = [
  "event",
  "server",
  "arguments",
  "callback",
  "status",
  "owner",
  "lastDelivery",
  "lastError",
  "expires",
] as const;

export function MCPEventSubscriptionsTab({ accessToken }: { accessToken: string | null }) {
  const { t } = useTranslation();
  const [offset, setOffset] = useQueryState("subscriptionOffset", offsetParser);
  const { data, error, isLoading, isFetching, refetch } = useQuery({
    queryKey: subscriptionKeys.list({ page: offset }),
    queryFn: () => fetchMCPEventSubscriptions(accessToken!, offset, 1),
    enabled: Boolean(accessToken),
    refetchInterval: 15000,
  });
  return (
    <section className="space-y-4 py-4" aria-label={t("mcpServers.eventSubscriptionsTab.regionLabel")}>
      <h2>{t("mcpServers.eventSubscriptionsTab.title")}</h2>
      <Button variant="outline" onClick={() => refetch()} disabled={isFetching}>
        {t("mcpServers.eventSubscriptionsTab.refresh")}
      </Button>
      {error && <p role="alert">{error.message}</p>}
      {isLoading && <p role="status">{t("mcpServers.eventSubscriptionsTab.loading")}</p>}
      {data?.subscriptions.length === 0 && <p>{t("mcpServers.eventSubscriptionsTab.empty")}</p>}
      {Boolean(data?.subscriptions.length) && (
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((column) => (
                <TableHead key={column}>{t(`mcpServers.eventSubscriptionsTab.columns.${column}`)}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {data?.subscriptions.map((row) => (
              <TableRow key={row.id}>
                <TableCell>{row.event_name}</TableCell>
                <TableCell>{row.server_id}</TableCell>
                <TableCell>{argumentsSummary(row.arguments)}</TableCell>
                <TableCell>{row.callback_host}</TableCell>
                <TableCell>{t(`mcpServers.eventSubscriptionsTab.status.${row.status}`)}</TableCell>
                <TableCell>{row.user_id ?? row.team_id ?? "-"}</TableCell>
                <TableCell>{row.last_delivery_at ?? "-"}</TableCell>
                <TableCell>
                  {row.last_error ? t(`mcpServers.eventSubscriptionsTab.failure.${row.last_error}`) : "-"}
                </TableCell>
                <TableCell>{row.refresh_before ?? t("mcpServers.eventSubscriptionsTab.noExpiry")}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      {offset > 0 && (
        <Button onClick={() => setOffset(Math.max(0, offset - 500))}>
          {t("mcpServers.eventSubscriptionsTab.previous")}
        </Button>
      )}
      {data?.next_offset != null && (
        <Button onClick={() => setOffset(data.next_offset!)}>{t("mcpServers.eventSubscriptionsTab.next")}</Button>
      )}
    </section>
  );
}
