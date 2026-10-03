"use client";

import { analysisKeysQuery, type Key } from "../../api/queries";

import { useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useLensApi } from "../../api/useLensApi";
import { AnalysisKeyDetails } from "./AnalysisKeyDetails";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";

export function AnalysisKeyPicker({
  accessToken,
  value,
  onChange,
}: {
  accessToken: string;
  value: string | null;
  onChange: (key: string | null) => void;
}) {
  const apiClient = useLensApi();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Key | null>(value ? { token: value } : null);

  const keyPages = useInfiniteQuery(analysisKeysQuery(apiClient, accessToken, query));
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
