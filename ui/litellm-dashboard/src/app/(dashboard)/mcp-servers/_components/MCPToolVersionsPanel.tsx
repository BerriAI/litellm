"use client";

import { type FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { mcpServersKeys } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import {
  clearMCPToolVersionDeprecation,
  getMCPToolVersions,
  pinMCPServerTools,
  setMCPToolVersionDeprecation,
  type MCPToolVersion,
} from "@/components/networking";

type MCPToolChangeKind = MCPToolVersion["change_kind"];

const CHANGE_KIND_LABELS: Record<MCPToolChangeKind, string> = {
  initial: "Initial",
  non_breaking: "Non-breaking",
  breaking: "Breaking",
  removed: "Removed",
};

const CHANGE_KIND_VARIANTS: Record<MCPToolChangeKind, "destructive" | "secondary" | "outline"> = {
  initial: "outline",
  non_breaking: "secondary",
  breaking: "destructive",
  removed: "destructive",
};

const TIMEZONE_SUFFIX = /(?:Z|[+-]\d{2}:?\d{2})$/i;

function formatUTCDate(value: string): string {
  const parsed: Date = new Date(value.includes("T") && !TIMEZONE_SUFFIX.test(value) ? `${value}Z` : value);
  return Number.isNaN(parsed.getTime()) ? value.slice(0, 10) : parsed.toISOString().slice(0, 10);
}

function ChangeKindBadge({ kind }: { kind: MCPToolChangeKind }) {
  return <Badge variant={CHANGE_KIND_VARIANTS[kind]}>{CHANGE_KIND_LABELS[kind]}</Badge>;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export interface MCPToolVersionsPanelProps {
  serverId: string;
  accessToken: string | null;
  isProxyAdmin: boolean;
  customHeaders?: Record<string, string>;
}

export function MCPToolVersionsPanel({
  serverId,
  accessToken,
  isProxyAdmin,
  customHeaders,
}: MCPToolVersionsPanelProps) {
  const queryClient = useQueryClient();
  const queryKey: readonly ["mcpToolVersions", string] = ["mcpToolVersions", serverId];
  const [changelog, setChangelog] = useState("");
  const [expandedTool, setExpandedTool] = useState<string | null>(null);
  const [editingVersion, setEditingVersion] = useState<string | null>(null);
  const [sunsetDate, setSunsetDate] = useState("");
  const [deprecationNote, setDeprecationNote] = useState("");
  const {
    data: versions,
    error: loadError,
    isLoading,
  } = useQuery({
    queryKey,
    queryFn: () => {
      if (!accessToken) throw new Error("An access token is required to load tool versions.");
      return getMCPToolVersions(accessToken, serverId);
    },
    enabled: Boolean(accessToken),
  });
  const invalidateVersions = async () => queryClient.invalidateQueries({ queryKey });
  const invalidatePinnedToolQueries = async (): Promise<void> => {
    await Promise.all([
      invalidateVersions(),
      queryClient.invalidateQueries({ queryKey: ["mcpTools", serverId] }),
      queryClient.invalidateQueries({ queryKey: mcpServersKeys.all }),
    ]);
  };
  const pinMutation = useMutation({
    mutationFn: () => {
      if (!accessToken) throw new Error("An access token is required to pin tools.");
      return pinMCPServerTools(accessToken, serverId, changelog, customHeaders);
    },
    onSuccess: invalidatePinnedToolQueries,
  });
  const deprecationMutation = useMutation({
    mutationFn: ({
      toolName,
      version,
      sunsetDate: value,
      note,
    }: {
      toolName: string;
      version: number;
      sunsetDate: string;
      note: string;
    }) => {
      if (!accessToken) throw new Error("An access token is required to update deprecation.");
      return setMCPToolVersionDeprecation(
        accessToken,
        { serverId, toolName, version },
        {
          sunset_date: value ? `${value}T00:00:00Z` : null,
          deprecation_note: note.trim() || null,
        },
      );
    },
    onSuccess: async () => {
      setEditingVersion(null);
      await invalidateVersions();
    },
  });
  const clearDeprecationMutation = useMutation({
    mutationFn: ({ toolName, version }: { toolName: string; version: number }) => {
      if (!accessToken) throw new Error("An access token is required to clear deprecation.");
      return clearMCPToolVersionDeprecation(accessToken, { serverId, toolName, version });
    },
    onSuccess: invalidateVersions,
  });
  const beginDeprecation = (version: MCPToolVersion): void => {
    setEditingVersion(`${version.tool_name}-${version.version}`);
    setSunsetDate(version.sunset_date ? formatUTCDate(version.sunset_date) : "");
    setDeprecationNote(version.deprecation_note ?? "");
  };
  const submitDeprecation = (event: FormEvent<HTMLFormElement>, toolName: string, version: number): void => {
    event.preventDefault();
    const variables = { toolName, version, sunsetDate, note: deprecationNote };
    deprecationMutation.mutate(variables);
  };

  if (isLoading) {
    return (
      <section aria-labelledby="mcp-tool-versions-title" className="mt-4">
        <h2 id="mcp-tool-versions-title" className="mb-3 text-lg font-semibold">
          Tool versions
        </h2>
        <div role="status" aria-label="Loading tool versions" className="space-y-3">
          <div className="space-y-2 rounded-lg border p-4">
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-3 w-2/3" />
          </div>
          <div className="space-y-2 rounded-lg border p-4">
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-3 w-2/3" />
          </div>
        </div>
      </section>
    );
  }

  if (!accessToken) {
    return (
      <section aria-labelledby="mcp-tool-versions-title" className="mt-4">
        <h2 id="mcp-tool-versions-title" className="mb-3 text-lg font-semibold">
          Tool versions
        </h2>
        <Alert variant="destructive">
          <AlertTitle>Could not load tool versions</AlertTitle>
          <AlertDescription>An access token is required to load tool versions.</AlertDescription>
        </Alert>
      </section>
    );
  }

  const orderedVersions: MCPToolVersion[] = versions ?? [];
  const toolNames: string[] = [...new Set(orderedVersions.map((version) => version.tool_name))].sort((a, b) =>
    a.localeCompare(b),
  );

  return (
    <section aria-labelledby="mcp-tool-versions-title" className="mt-4 space-y-4">
      <h2 id="mcp-tool-versions-title" className="text-lg font-semibold">
        Tool versions
      </h2>

      {isProxyAdmin && (
        <Card className="gap-3 p-4">
          <label htmlFor="mcp-tool-version-changelog" className="text-sm font-medium">
            Changelog
          </label>
          <Textarea
            id="mcp-tool-version-changelog"
            value={changelog}
            onChange={(event) => setChangelog(event.currentTarget.value)}
            rows={3}
          />
          <p className="text-xs text-muted-foreground">
            tools/list serves the pinned tool list until the next pin. Each changed tool gets a new version.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Button type="button" disabled={pinMutation.isPending} onClick={() => pinMutation.mutate()}>
              {pinMutation.isPending ? "Pinning..." : "Pin and record versions"}
            </Button>
            {pinMutation.error && (
              <p role="alert" className="text-sm text-destructive">
                {errorMessage(pinMutation.error)}
              </p>
            )}
          </div>
        </Card>
      )}

      {deprecationMutation.error && (
        <Alert variant="destructive">
          <AlertTitle>Could not update deprecation</AlertTitle>
          <AlertDescription>{errorMessage(deprecationMutation.error)}</AlertDescription>
        </Alert>
      )}
      {clearDeprecationMutation.error && (
        <Alert variant="destructive">
          <AlertTitle>Could not remove deprecation</AlertTitle>
          <AlertDescription>{errorMessage(clearDeprecationMutation.error)}</AlertDescription>
        </Alert>
      )}

      {loadError && (
        <Alert variant="destructive">
          <AlertTitle>Could not load tool versions</AlertTitle>
          <AlertDescription>{errorMessage(loadError)}</AlertDescription>
        </Alert>
      )}

      {!loadError && orderedVersions.length === 0 && (
        <div className="rounded-lg border border-dashed border-border bg-card p-8 text-center">
          <p className="text-sm text-muted-foreground">
            No versions recorded yet. Pin the tool list to start tracking versions.
          </p>
        </div>
      )}

      {!loadError && orderedVersions.length > 0 && (
        <div className="space-y-3">
          {toolNames.map((toolName) => {
            const history: MCPToolVersion[] = orderedVersions
              .filter((version) => version.tool_name === toolName)
              .sort((left, right) => right.version - left.version);
            const latest: MCPToolVersion | undefined = history[0];
            if (!latest) return null;
            const isExpanded: boolean = expandedTool === toolName;

            return (
              <Card key={toolName} aria-label={`${toolName} tool versions`} className="gap-3 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="font-medium">{toolName}</h3>
                    <Badge variant="outline">v{latest.version}</Badge>
                    <ChangeKindBadge kind={latest.change_kind} />
                    {latest.deprecated_at && (
                      <Badge variant="destructive">
                        Deprecated
                        {latest.sunset_date ? `, sunset ${formatUTCDate(latest.sunset_date)}` : ""}
                      </Badge>
                    )}
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    aria-expanded={isExpanded}
                    aria-controls={`mcp-tool-history-${toolName}`}
                    onClick={() => setExpandedTool(isExpanded ? null : toolName)}
                  >
                    {isExpanded ? `Hide history for ${toolName}` : `Show history for ${toolName}`}
                  </Button>
                </div>

                {isExpanded && (
                  <div id={`mcp-tool-history-${toolName}`} className="space-y-3 border-t pt-3">
                    {history.map((version) => {
                      const versionKey: string = `${toolName}-${version.version}`;
                      return (
                        <article key={versionKey} className="space-y-2 rounded-md border p-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <h4 className="font-medium">v{version.version}</h4>
                            <ChangeKindBadge kind={version.change_kind} />
                            <span className="text-xs text-muted-foreground">
                              Created {formatUTCDate(version.created_at)}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              Created by {version.created_by ?? "Unknown"}
                            </span>
                          </div>

                          {version.changes.length > 0 && (
                            <ul className="space-y-1 text-sm">
                              {version.changes.map((change) => (
                                <li
                                  key={`${versionKey}-${change.summary}`}
                                  className="flex flex-wrap items-center gap-2"
                                >
                                  <span>{change.summary}</span>
                                  {change.breaking && <Badge variant="destructive">Breaking</Badge>}
                                </li>
                              ))}
                            </ul>
                          )}

                          {version.changelog && (
                            <p className="text-sm">
                              Changelog: <span className="text-muted-foreground">{version.changelog}</span>
                            </p>
                          )}
                          {version.deprecated_at && (
                            <div className="space-y-1 text-sm">
                              <p>Deprecated {formatUTCDate(version.deprecated_at)}</p>
                              {version.sunset_date && <p>Sunset {formatUTCDate(version.sunset_date)}</p>}
                              {version.deprecation_note && <p>{version.deprecation_note}</p>}
                            </div>
                          )}

                          {isProxyAdmin && (
                            <div className="space-y-3">
                              {version.deprecated_at ? (
                                <div className="flex flex-wrap gap-3">
                                  <Button type="button" variant="outline" onClick={() => beginDeprecation(version)}>
                                    Edit deprecation for {toolName} v{version.version}
                                  </Button>
                                  <Button
                                    type="button"
                                    variant="outline"
                                    disabled={clearDeprecationMutation.isPending}
                                    onClick={() =>
                                      clearDeprecationMutation.mutate({ toolName, version: version.version })
                                    }
                                  >
                                    Remove deprecation for {toolName} v{version.version}
                                  </Button>
                                </div>
                              ) : (
                                <Button type="button" variant="outline" onClick={() => beginDeprecation(version)}>
                                  Deprecate {toolName} v{version.version}
                                </Button>
                              )}

                              {editingVersion === versionKey && (
                                <form
                                  className="grid gap-3 sm:max-w-md"
                                  onSubmit={(event) => submitDeprecation(event, toolName, version.version)}
                                >
                                  <label htmlFor={`sunset-date-${versionKey}`} className="text-sm font-medium">
                                    Sunset date
                                  </label>
                                  <Input
                                    id={`sunset-date-${versionKey}`}
                                    type="date"
                                    value={sunsetDate}
                                    onChange={(event) => setSunsetDate(event.currentTarget.value)}
                                  />
                                  <label htmlFor={`deprecation-note-${versionKey}`} className="text-sm font-medium">
                                    Note
                                  </label>
                                  <Input
                                    id={`deprecation-note-${versionKey}`}
                                    value={deprecationNote}
                                    onChange={(event) => setDeprecationNote(event.currentTarget.value)}
                                  />
                                  <Button type="submit" disabled={deprecationMutation.isPending}>
                                    {deprecationMutation.isPending ? "Saving..." : "Save"}
                                  </Button>
                                </form>
                              )}
                            </div>
                          )}
                        </article>
                      );
                    })}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </section>
  );
}
