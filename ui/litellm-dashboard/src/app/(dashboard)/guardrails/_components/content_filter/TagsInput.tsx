"use client";

import React, { useState } from "react";

import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";

export interface TagsInputOption {
  label: string;
  value: string;
}

interface TagsInputProps {
  value: string[];
  onValueChange: (value: string[]) => void;
  options?: TagsInputOption[];
  placeholder?: string;
  emptyText?: string;
  tokenSeparators?: string[];
  loading?: boolean;
  disabled?: boolean;
  id?: string;
}

const splitOnSeparators = (raw: string, separators: string[]): string[] =>
  separators.reduce<string[]>((parts, separator) => parts.flatMap((part) => part.split(separator)), [raw]);

const toOption = (options: TagsInputOption[], value: string): TagsInputOption =>
  options.find((option) => option.value === value) ?? { label: value, value };

const matchesQuery = (option: TagsInputOption, query: string): boolean =>
  option.label.toLowerCase().includes(query.trim().toLowerCase());

export const TagsInput = ({
  value,
  onValueChange,
  options = [],
  placeholder,
  emptyText = "No matching options",
  tokenSeparators = [],
  loading = false,
  disabled = false,
  id,
}: TagsInputProps) => {
  const anchor = useComboboxAnchor();
  const [query, setQuery] = useState("");

  const selected = value.map((tag) => toOption(options, tag));
  const pending = query.trim();
  const isCreatable = pending.length > 0 && !options.some((option) => option.value === pending);
  const items = isCreatable ? [{ label: pending, value: pending }, ...options] : options;

  const addTags = (tags: string[]) => {
    const additions = tags
      .map((tag) => tag.trim())
      .filter(Boolean)
      .filter((tag, index, all) => all.indexOf(tag) === index && !value.includes(tag));
    if (additions.length > 0) onValueChange([...value, ...additions]);
  };

  const commitPending = () => {
    setQuery("");
    addTags([query]);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    if (event.currentTarget.getAttribute("aria-activedescendant")) return;
    commitPending();
  };

  const handleInputValueChange = (next: string) => {
    if (!tokenSeparators.some((separator) => next.includes(separator))) {
      setQuery(next);
      return;
    }
    const parts = splitOnSeparators(next, tokenSeparators);
    setQuery(parts[parts.length - 1] ?? "");
    addTags(parts.slice(0, -1));
  };

  return (
    <Combobox
      multiple
      items={items}
      value={selected}
      onValueChange={(next: TagsInputOption[]) => {
        setQuery("");
        onValueChange(next.map((option) => option.value));
      }}
      inputValue={query}
      onInputValueChange={handleInputValueChange}
      isItemEqualToValue={(option: TagsInputOption, other: TagsInputOption) => option.value === other.value}
      itemToStringLabel={(option: TagsInputOption) => option.label}
      filter={matchesQuery}
      openOnInputClick
      disabled={disabled || loading}
    >
      <ComboboxChips render={<div ref={anchor} />} className="min-h-8 py-1 text-sm">
        <ComboboxValue>
          {(chips: TagsInputOption[]) => (
            <>
              {chips.map((option) => (
                <ComboboxChip key={option.value} aria-label={option.label}>
                  {option.label}
                </ComboboxChip>
              ))}
              <ComboboxChipsInput
                id={id}
                placeholder={loading ? "Loading..." : placeholder}
                className="min-w-24"
                onBlur={commitPending}
                onKeyDown={handleKeyDown}
              />
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{emptyText}</ComboboxEmpty>
        <ComboboxList>
          {(option: TagsInputOption) => (
            <ComboboxItem key={option.value} value={option} title={option.label}>
              {option.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
};
