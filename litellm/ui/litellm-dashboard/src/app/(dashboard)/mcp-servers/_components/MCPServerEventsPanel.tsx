"use client";

import React from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { fetchMCPEventTypes } from "@/components/networking";
import type { MCPEventType } from "@/components/mcp_tools/types";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";

const eventTypeKeys = createQueryKeys("mcpEventTypes");

interface MCPServerEventsPanelProps {
  serverId: string;
  accessToken: string | null;
}

export function MCPServerEventsPanel({ serverId, accessToken }: MCPServerEventsPanelProps) {
  const { t } = useTranslation();
  const { data, error, isLoading } = useQuery<MCPEventType[], Error>({
    queryKey: eventTypeKeys.detail(serverId),
    queryFn: () => fetchMCPEventTypes(accessToken!, serverId),
    enabled: !!accessToken,
  });
  if (isLoading) {
    return (
      <div role="status" className="flex items-center gap-3 p-6">
        <UiLoadingSpinner className="size-5 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">{t("mcpServers.serverView.events.loading")}</p>
      </div>
    );
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>{t("mcpServers.serverView.events.loadError")}</AlertTitle>
        <AlertDescription>{error.message}</AlertDescription>
      </Alert>
    );
  }
  if (!data || data.length === 0) {
    return <p className="p-6 text-sm text-muted-foreground">{t("mcpServers.serverView.events.empty")}</p>;
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>{t("mcpServers.serverView.events.columns.name")}</TableHead>
          <TableHead>{t("mcpServers.serverView.events.columns.description")}</TableHead>
          <TableHead>{t("mcpServers.serverView.events.columns.delivery")}</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.map((eventType) => (
          <TableRow key={eventType.name}>
            <TableCell className="font-mono text-xs">{eventType.name}</TableCell>
            <TableCell className="text-sm">{eventType.description ?? "-"}</TableCell>
            <TableCell className="space-x-1">
              {eventType.delivery.map((mode) => (
                <Badge key={mode} variant="outline">
                  {mode}
                </Badge>
              ))}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default MCPServerEventsPanel;
