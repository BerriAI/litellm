"use client";

import { analysisKeysQuery, type Key } from "../../api/queries";

import { Controller, useFormContext, useWatch } from "react-hook-form";
import { useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useLensApi } from "../../api/useLensApi";
import { AnalysisKeyDetails } from "./AnalysisKeyDetails";
import type { WorkerFormInput } from "./workerSchema";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";

export function AnalysisKeyPicker({ accessToken }: { accessToken: string }) {
  const { control } = useFormContext<WorkerFormInput>();
  const value = useWatch({ control, name: "analysisKey" });
  const apiClient = useLensApi();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Key | null>(value ? { token: value } : null);

  const keyPages = useInfiniteQuery(analysisKeysQuery(apiClient, accessToken, query));
  const keys = keyPages.data?.pages.flatMap((page) => page.keys) ?? [];
  const choice = keys.find((key) => key.token === value) ?? (selected?.token === value ? selected : null);
  const loading = keyPages.isFetching;

  const choices = choice && !keys.some((key) => key.token === choice.token) ? [choice, ...keys] : keys;
  const items = keyPages.hasNextPage
    ? [...choices, { token: "load-more", key_alias: loading ? "Loading…" : "Load more keys" }]
    : choices;
  return (
    <div className="space-y-2">
      <p className="text-sm">Charge analysis to</p>
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1">
          <Controller
            control={control}
            name="analysisKey"
            render={({ field }) => {
              const handleValueChange = (key: Key | null, details: { cancel: () => void }) => {
                if (key?.token === "load-more") {
                  details.cancel();
                  if (!loading) void keyPages.fetchNextPage();
                  return;
                }
                setSelected(key);
                field.onChange(key?.token ?? null);
              };
              return (
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
                  onValueChange={handleValueChange}
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
              );
            }}
          />
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
