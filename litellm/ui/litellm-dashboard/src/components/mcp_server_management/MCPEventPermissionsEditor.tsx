"use client";

import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useQueries } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { fetchMCPEventTypes } from "@/components/networking";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { type EventPermissions, sameEventPermissions, withServerEvents } from "./eventPermissions";

const eventTypeKeys = createQueryKeys("mcpEventTypes");

interface MCPEventPermissionsEditorProps {
  accessToken: string | null;
  serverIds: readonly string[];
  value: EventPermissions;
  disabled?: boolean;
  onSave: (next: Record<string, string[]>) => Promise<void>;
}

export function MCPEventPermissionsEditor({
  accessToken,
  serverIds,
  value,
  disabled = false,
  onSave,
}: MCPEventPermissionsEditorProps) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<EventPermissions>(value);
  const [saving, setSaving] = useState(false);
  const queries = useQueries({
    queries: serverIds.map((serverId) => ({
      queryKey: eventTypeKeys.detail(serverId),
      queryFn: () => fetchMCPEventTypes(accessToken!, serverId),
      enabled: !!accessToken,
      retry: false,
    })),
  });
  const servers = serverIds
    .map((serverId, index) => ({ serverId, types: queries[index]?.data ?? [] }))
    .filter((server) => server.types.length > 0);
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
    <section aria-label={t("mcpServerManagement.eventPermissions.title")} className="space-y-3">
      <div>
        <h4 className="text-sm font-medium text-foreground">{t("mcpServerManagement.eventPermissions.title")}</h4>
        <p className="text-xs text-muted-foreground">{t("mcpServerManagement.eventPermissions.description")}</p>
      </div>
      {servers.map(({ serverId, types }) => (
        <div key={serverId} className="space-y-1">
          <span className="font-mono text-xs text-foreground">{serverId}</span>
          <MultiSelect
            value={[...(draft[serverId] ?? [])]}
            onValueChange={(selected) => setDraft(withServerEvents(draft, serverId, selected))}
            options={types.map((eventType) => ({ label: eventType.name, value: eventType.name }))}
            placeholder={t("mcpServerManagement.eventPermissions.allEvents")}
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
