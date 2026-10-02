"use client";

import { useState } from "react";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Input } from "@/components/ui/input";
import { AnalysisKeyDetails } from "./AnalysisKeyDetails";
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
}: {
  accessToken: string;
  value: string | null;
  onChange: (key: string | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Key | null>(value ? { token: value } : null);

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
      <div className="flex flex-wrap items-start gap-2">
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
      </div>
      {choice && <AnalysisKeyDetails accessToken={accessToken} keyId={choice.token} />}
      {keyPages.error && (
        <p role="alert" className="text-sm text-destructive">
          {keyPages.error.message}
        </p>
      )}
    </div>
  );
}

export type AnalysisAccess = { model: string | null; budget: string };

export function AnalysisAccessFields({
  accessToken,
  value,
  onChange,
}: {
  accessToken: string;
  value: AnalysisAccess;
  onChange: (value: AnalysisAccess) => void;
}) {
  const models = useQuery({
    queryKey: ["lens-models", accessToken],
    queryFn: () => apiClient.get<{ data: { id: string }[] }>("/models", { accessToken }),
  });
  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <label htmlFor="analysis-access-model" className="block text-sm font-medium">
          Analysis model
        </label>
        <SearchSelect
          inputId="analysis-access-model"
          options={(models.data?.data ?? []).map(({ id }) => ({ label: id, value: id }))}
          value={value.model}
          onValueChange={(model) => onChange({ ...value, model })}
          placeholder={models.isLoading ? "Loading models…" : "Select a model"}
        />
      </div>
      <div className="space-y-2">
        <label htmlFor="analysis-access-budget" className="block text-sm font-medium">
          Monthly limit (USD)
        </label>
        <Input
          id="analysis-access-budget"
          type="number"
          min="0.01"
          step="0.01"
          value={value.budget}
          onChange={(e) => onChange({ ...value, budget: e.target.value })}
        />
        <p className="text-xs text-muted-foreground">Shared across all investigations.</p>
      </div>
      {models.error && (
        <p role="alert" className="text-sm text-destructive">
          {models.error.message}
        </p>
      )}
    </div>
  );
}

export async function createAnalysisKey(accessToken: string, access: AnalysisAccess): Promise<string> {
  if (!access.model || !Number.isFinite(Number(access.budget)) || Number(access.budget) <= 0)
    throw new Error("Choose a model and a monthly limit greater than zero");
  const result = await apiClient.post("/key/generate", {
    accessToken,
    body: {
      key_alias: "Lens analysis",
      models: [access.model],
      max_budget: Number(access.budget),
      budget_duration: "1mo",
      metadata: { purpose: "lens" },
    },
  });
  if (!result.token_id) throw new Error("The proxy did not return the new key's ID");
  return result.token_id;
}
