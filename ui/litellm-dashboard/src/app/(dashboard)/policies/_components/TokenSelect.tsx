"use client";

import * as React from "react";

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

interface TokenSelectProps {
  id: string;
  value: readonly string[] | undefined;
  onValueChange: (value: string[]) => void;
  onBlur?: () => void;
  placeholder: string;
  options?: readonly string[];
  allowCustomValues?: boolean;
  tokenSeparators?: readonly string[];
  emptyText?: string;
  ariaInvalid?: true;
  ariaDescribedBy?: string;
}

const splitOnSeparators = (text: string, separators: readonly string[]): string[] =>
  separators.reduce<string[]>((parts, separator) => parts.flatMap((part) => part.split(separator)), [text]);

const withAdditions = (current: readonly string[], additions: readonly string[]): string[] => [
  ...current,
  ...additions.filter((addition) => addition !== "" && !current.includes(addition)),
];

export const includesQuery = (item: string, query: string): boolean => item.toLowerCase().includes(query.toLowerCase());

export const TokenSelect: React.FC<TokenSelectProps> = ({
  id,
  value,
  onValueChange,
  onBlur,
  placeholder,
  options,
  allowCustomValues = false,
  tokenSeparators = [],
  emptyText = "No options found",
  ariaInvalid,
  ariaDescribedBy,
}) => {
  const anchor = useComboboxAnchor();
  const [query, setQuery] = React.useState("");
  const selected = value ?? [];
  const showDropdown = options !== undefined;

  const pendingCustomValue = allowCustomValues && query.trim() !== "" && !options?.includes(query.trim());
  const items = pendingCustomValue ? [...(options ?? []), query.trim()] : options ?? [];

  const handleInputValueChange = (next: string) => {
    if (!allowCustomValues || !tokenSeparators.some((separator) => next.includes(separator))) {
      setQuery(next);
      return;
    }
    const parts = splitOnSeparators(next, tokenSeparators);
    const committed = parts.slice(0, -1).map((part) => part.trim());
    onValueChange(withAdditions(selected, committed));
    setQuery(parts[parts.length - 1]);
  };

  const handleBlur = () => {
    const pending = query.trim();
    if (allowCustomValues && pending !== "") {
      onValueChange(withAdditions(selected, [pending]));
    }
    setQuery("");
    onBlur?.();
  };

  return (
    <Combobox
      multiple
      autoHighlight={showDropdown}
      open={showDropdown ? undefined : false}
      items={items}
      value={selected as string[]}
      onValueChange={(next: string[]) => {
        onValueChange(next);
        setQuery("");
      }}
      inputValue={query}
      onInputValueChange={handleInputValueChange}
      filter={includesQuery}
    >
      <ComboboxChips render={<div ref={anchor} />}>
        <ComboboxValue>
          {(chips: string[]) => (
            <>
              {chips.map((chip) => (
                <ComboboxChip key={chip} aria-label={chip}>
                  {chip}
                </ComboboxChip>
              ))}
              <ComboboxChipsInput
                id={id}
                placeholder={placeholder}
                aria-invalid={ariaInvalid}
                aria-describedby={ariaDescribedBy}
                onBlur={handleBlur}
              />
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      {showDropdown && (
        <ComboboxContent anchor={anchor}>
          <ComboboxEmpty>{emptyText}</ComboboxEmpty>
          <ComboboxList>
            {(item: string) => (
              <ComboboxItem key={item} value={item} title={item}>
                {item}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      )}
    </Combobox>
  );
};
