"use client";

import { useInfiniteKeyAliases } from "@/app/(dashboard)/hooks/keys/useKeyAliases";
import { Combobox, type ComboboxOption } from "@/components/shared/Combobox";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useMemo, useState } from "react";

interface KeyAliasComboboxProps {
  value?: string;
  onValueChange: (value: string) => void;
  teamId?: string;
  placeholder?: string;
}

const PAGE_SIZE = 50;

export function KeyAliasCombobox({
  value,
  onValueChange,
  teamId,
  placeholder = "Select Key Alias…",
}: KeyAliasComboboxProps) {
  const [search, setSearch] = useState("");
  const [debouncedSearch] = useDebouncedValue(search, { wait: 300 });

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteKeyAliases(
    PAGE_SIZE,
    debouncedSearch || undefined,
    teamId,
  );

  const options = useMemo<ComboboxOption[]>(() => {
    const seen = new Set<string>();
    const result: ComboboxOption[] = [];
    for (const page of data?.pages ?? []) {
      for (const alias of page.aliases) {
        if (!alias || seen.has(alias)) continue;
        seen.add(alias);
        result.push({ label: alias, value: alias });
      }
    }
    return result;
  }, [data]);

  return (
    <Combobox
      options={options}
      value={value}
      onValueChange={onValueChange}
      placeholder={placeholder}
      searchPlaceholder="Search key aliases…"
      emptyText="No key aliases found"
      loading={isLoading || isFetchingNextPage}
      searchValue={search}
      onSearchChange={setSearch}
      onReachEnd={() => {
        if (hasNextPage && !isFetchingNextPage) {
          fetchNextPage();
        }
      }}
    />
  );
}
