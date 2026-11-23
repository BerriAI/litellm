"use client";

import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useQueries } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { fetchMCPEventTypes } from "@/components/networking";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import {
  type EventPermissions,
  eventServers,
  sameEventPermissions,
  toggleServerBlocked,
  withServerEvents,
} from "./eventPermissions";

const eventTypeKeys = createQueryKeys("mcpEventTypes");

interface MCPEventPermissionsEditorProps {
  accessToken: string | null;
  value: EventPermissions;
  disabled?: boolean;
  className?: string;
  onSave: (next: Record<string, string[]>) => Promise<void>;
}

export function MCPEventPermissionsEditor({
  accessToken,
  value,
  disabled = false,
  className = "",
  onSave,
}: MCPEventPermissionsEditorProps) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<EventPermissions>(value);
  const [saving, setSaving] = useState(false);
  const { data: allServers } = useMCPServers();
  const candidates = eventServers(allServers ?? []);
  const queries = useQueries({
    queries: candidates.map(({ serverId }) => ({
      queryKey: eventTypeKeys.detail(serverId),
      queryFn: () => fetchMCPEventTypes(accessToken!, serverId),
      enabled: !!accessToken,
      retry: false,
    })),
  });
  const servers = [
    ...candidates.map((candidate, index) => ({
      ...candidate,
      names: (queries[index]?.data ?? []).map(({ name }) => name),
    })),
    ...Object.keys(value)
      .filter((serverId) => !candidates.some((candidate) => candidate.serverId === serverId))
      .map((serverId) => ({ serverId, label: serverId, names: [] })),
  ];
  const loading = queries.some((query) => query.isLoading);
  const failed = queries.filter((query) => query.isError);
  if (servers.length === 0) return null;
  const dirty = !sameEventPermissions(draft, value);
  const save = async () => {
    setSaving(true);
    try {
      await onSave(Object.fromEntries(Object.entries(draft).map(([serverId, events]) => [serverId, [...events]])));
    } finally {
      setSaving(false);
    }
  };
  return (
    <section aria-label={t("mcpServerManagement.eventPermissions.title")} className={`space-y-3 ${className}`}>
      <div>
        <h4 className="text-sm font-medium text-foreground">{t("mcpServerManagement.eventPermissions.title")}</h4>
        <p className="text-xs text-muted-foreground">{t("mcpServerManagement.eventPermissions.description")}</p>
      </div>
      {loading ? (
        <p className="text-xs text-muted-foreground">{t("mcpServerManagement.eventPermissions.loading")}</p>
      ) : null}
      {failed.length > 0 ? (
        <div role="alert" className="flex items-center gap-2 text-xs text-destructive">
          {t("mcpServerManagement.eventPermissions.loadError")}
          <Button size="sm" variant="outline" onClick={() => failed.forEach((query) => void query.refetch())}>
            {t("mcpServerManagement.eventPermissions.retry")}
          </Button>
        </div>
      ) : null}
      {servers.map(({ serverId, label, names }) => (
        <div key={serverId} className="space-y-1">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-foreground">{label}</span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setDraft(toggleServerBlocked(draft, serverId))}
              disabled={disabled}
            >
              {t(
                draft[serverId]?.length === 0
                  ? "mcpServerManagement.eventPermissions.allowAll"
                  : "mcpServerManagement.eventPermissions.blockAll",
              )}
            </Button>
          </div>
          <MultiSelect
            allowCustomValues="literal"
            value={[...(draft[serverId] ?? [])]}
            onValueChange={(selected) => setDraft(withServerEvents(draft, serverId, selected))}
            options={Array.from(new Set([...names, ...(draft[serverId] ?? [])]), (name) => ({
              label: name,
              value: name,
            }))}
            placeholder={t(
              serverId in draft
                ? "mcpServerManagement.eventPermissions.noEvents"
                : "mcpServerManagement.eventPermissions.allEvents",
            )}
            disabled={disabled}
            className="rounded-lg"
          />
        </div>
      ))}
      <Button size="sm" onClick={save} disabled={disabled || !dirty || saving}>
        {t("mcpServerManagement.eventPermissions.save")}
      </Button>
    </section>
  );
}

export default MCPEventPermissionsEditor;
