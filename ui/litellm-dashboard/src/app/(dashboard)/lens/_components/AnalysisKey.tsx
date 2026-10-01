"use client";

import { useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";

const keySchema = z.object({ token: z.string(), key_alias: z.string().nullable().optional() });
const pageSchema = z.object({ keys: z.array(keySchema), total_pages: z.number() });
type Key = z.infer<typeof keySchema>;

export function AnalysisKey({
  accessToken,
  value,
  onChange,
  name,
}: {
  accessToken: string;
  value: string | null;
  onChange: (key: string | null) => void;
  name: string;
}) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Key | null>(value ? { token: value } : null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const queryOptions = {
    queryKey: ["lens-analysis-keys", accessToken, query],
    initialPageParam: 1,
    queryFn: async ({ pageParam, signal }: { pageParam: number; signal: AbortSignal }) =>
      pageSchema.parse(
        await apiClient.get("/key/list", {
          accessToken,
          signal,
          query: {
            page: String(pageParam),
            size: "25",
            return_full_object: "true",
            key_alias: query || undefined,
            substring_matching: "true",
            include_team_keys: "true",
            include_created_by_keys: "true",
            status: "active",
          },
        }),
      ),
    getNextPageParam: (lastPage: z.infer<typeof pageSchema>, pages: z.infer<typeof pageSchema>[]) =>
      pages.length < lastPage.total_pages ? pages.length + 1 : undefined,
  };
  const keyPages = useInfiniteQuery(queryOptions);
  const keys = keyPages.data?.pages.flatMap((page) => page.keys) ?? [];
  const choice = keys.find((key) => key.token === value) ?? selected;
  const loading = keyPages.isFetching;

  const create = async () => {
    setCreating(true);
    setError("");
    try {
      const result = await apiClient.post("/key/generate", {
        accessToken,
        body: {
          key_alias: `Lens: ${name}`,
          models: [],
          metadata: { purpose: "lens" },
        },
      });
      if (!result.token_id) throw new Error("The proxy did not return the new key's ID");
      const key = { token: result.token_id, key_alias: `Lens: ${name}` };
      setSelected(key);
      onChange(key.token);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not create a key");
    } finally {
      setCreating(false);
    }
  };
  const changeKey = (key: Key | null, details: { cancel: () => void }) => {
    if (key?.token === "load-more") {
      details.cancel();
      if (!loading) void keyPages.fetchNextPage();
      return;
    }
    setSelected(key);
    onChange(key?.token ?? null);
  };
  const choices = choice && !keys.some((key) => key.token === choice.token) ? [choice, ...keys] : keys;
  const items = keyPages.hasNextPage
    ? [...choices, { token: "load-more", key_alias: loading ? "Loading…" : "Load more keys" }]
    : choices;
  return (
    <div className="space-y-2">
      <p className="text-sm">Charge analysis to</p>
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <Combobox
            items={items}
            value={choice}
            filter={null}
            itemToStringLabel={(key: Key) => key.key_alias || `${key.token.slice(0, 8)}…`}
            isItemEqualToValue={(a: Key, b: Key) => a.token === b.token}
            onInputValueChange={(text, details) => {
              if (details.reason === "input-change" || details.reason === "input-clear") {
                setQuery(text);
              }
            }}
            onValueChange={changeKey}
          >
            <ComboboxInput aria-label="Charge analysis to" placeholder="Search existing keys" />
            <ComboboxContent>
              <ComboboxEmpty>{loading ? "Loading keys…" : "No matching keys"}</ComboboxEmpty>
              <ComboboxList>
                {(key: Key) => (
                  <ComboboxItem key={key.token} value={key}>
                    {key.key_alias || `${key.token.slice(0, 8)}…`}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
        <Button variant="outline" disabled={creating} onClick={() => void create()}>
          {creating ? "Creating…" : "Create worker key"}
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Spend appears under this key in API Keys. Its permissions and limits apply.
      </p>
      {(error || keyPages.error) && (
        <p role="alert" className="text-sm text-destructive">
          {error || keyPages.error?.message}
        </p>
      )}
    </div>
  );
}
