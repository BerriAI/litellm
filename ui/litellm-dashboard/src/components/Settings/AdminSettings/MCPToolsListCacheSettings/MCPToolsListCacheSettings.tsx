"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { toast } from "@/lib/toast";
import {
  useMCPToolsListCacheSettings,
  useUpdateMCPToolsListCacheSettings,
} from "@/app/(dashboard)/hooks/mcpToolsListCacheSettings/useMCPToolsListCacheSettings";

interface MCPToolsListCacheSettingsProps {
  accessToken: string | null;
}

export default function MCPToolsListCacheSettings({ accessToken }: MCPToolsListCacheSettingsProps) {
  const { data, isLoading, isError, error } = useMCPToolsListCacheSettings();
  const { mutate: updateSettings, isPending } = useUpdateMCPToolsListCacheSettings();
  const [ttlSeconds, setTtlSeconds] = useState("0");

  useEffect(() => {
    const storedValue = data?.values.ttl_seconds;
    if (storedValue !== undefined) setTtlSeconds(String(storedValue));
  }, [data?.values.ttl_seconds]);

  if (!accessToken)
    return <div className="p-6 text-center text-muted-foreground">Please log in to configure caching.</div>;
  if (isLoading) return <div className="p-6 text-muted-foreground">Loading cache settings...</div>;
  if (isError)
    return (
      <div className="p-6 text-destructive">
        {error instanceof Error ? error.message : "Could not load cache settings"}
      </div>
    );

  return (
    <Card>
      <CardHeader>
        <CardTitle>MCP tools/list cache</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-sm text-muted-foreground">
          Cache upstream tool lists for this many seconds. Set to 0 to disable the cache.
        </p>
        <label className="flex flex-col gap-2 text-sm" htmlFor="mcp-tools-list-cache-ttl">
          Cache TTL in seconds
          <Input
            id="mcp-tools-list-cache-ttl"
            type="number"
            min="0"
            step="any"
            value={ttlSeconds}
            onChange={(event) => setTtlSeconds(event.target.value)}
            disabled={isPending}
          />
        </label>
        <Button
          type="button"
          disabled={isPending || ttlSeconds === "" || !Number.isFinite(Number(ttlSeconds)) || Number(ttlSeconds) < 0}
          onClick={() => {
            updateSettings(
              { ttl_seconds: Number(ttlSeconds) },
              {
                onSuccess: () => toast.success("Cache settings updated successfully."),
                onError: (saveError) => toast.fromError(saveError),
              },
            );
          }}
        >
          Save settings
        </Button>
      </CardContent>
    </Card>
  );
}
