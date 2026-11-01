"use client";

import { useState, type ReactNode } from "react";
import { Plus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Combobox,
  ComboboxCollection,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxGroup,
  ComboboxInput,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
  ComboboxTrigger,
} from "@/components/ui/combobox";
import { InputGroupButton } from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/cva.config";
import type { PickerOption } from "./usePlaygroundCatalog";

export type OptionGroup = Readonly<{ label: string; options: readonly PickerOption[] }>;

type GroupItems = Readonly<{ value: string; items: readonly string[] }>;

const toItems = (groups: readonly OptionGroup[]): GroupItems[] =>
  groups
    .filter((group) => group.options.length > 0)
    .map((group) => ({ value: group.label, items: group.options.map((option) => option.value) }));

const isListed = (index: ReadonlyMap<string, PickerOption>, name: string): boolean =>
  index.has(name) || [...index.values()].some((option) => option.label === name);

const optionIndex = (groups: readonly OptionGroup[]): ReadonlyMap<string, PickerOption> =>
  new Map(groups.flatMap((group) => group.options.map((option) => [option.value, option] as const)));

function GroupedOptions({ index, mono = false }: { index: ReadonlyMap<string, PickerOption>; mono?: boolean }) {
  return (
    <ComboboxList>
      {(group: GroupItems) => (
        <ComboboxGroup key={group.value} items={[...group.items]}>
          <ComboboxLabel>{group.value}</ComboboxLabel>
          <ComboboxCollection>
            {(value: string) => {
              const option = index.get(value);
              if (!option) return <CustomItem key={value} value={value} />;
              return (
                <ComboboxItem key={value} value={value} className={cn("items-start", mono && "font-mono text-xs")}>
                  <span className="flex min-w-0 flex-col">
                    <span className="truncate">{option.label}</span>
                    {option.description && (
                      <span className="truncate font-sans text-xs text-muted-foreground">{option.description}</span>
                    )}
                  </span>
                </ComboboxItem>
              );
            }}
          </ComboboxCollection>
        </ComboboxGroup>
      )}
    </ComboboxList>
  );
}

function CustomItem({ value }: { value: string }) {
  return (
    <ComboboxItem value={value} className="font-mono text-xs">
      <Plus />
      <span className="min-w-0 truncate">
        <span className="font-sans">Use</span> {value}
      </span>
    </ComboboxItem>
  );
}

function CountBadge({ count }: { count: number }) {
  if (count === 0) return null;
  return (
    <Badge variant="secondary" className="h-4 min-w-4 px-1 tabular-nums">
      {count}
    </Badge>
  );
}

export function ChecklistMenu({
  label,
  icon,
  groups,
  value,
  onValueChange,
  isLoading,
  emptyText,
  defaultOpen = false,
}: {
  label: string;
  icon: ReactNode;
  groups: readonly OptionGroup[];
  value: readonly string[];
  onValueChange: (value: readonly string[]) => void;
  isLoading: boolean;
  emptyText: string;
  defaultOpen?: boolean;
}) {
  const index = optionIndex(groups);
  const items = toItems(groups);
  return (
    <Combobox
      items={items}
      multiple
      value={[...value]}
      onValueChange={(next: string[]) => onValueChange(next)}
      itemToStringLabel={(item: string) => index.get(item)?.label ?? item}
      defaultOpen={defaultOpen}
      autoHighlight
    >
      <ComboboxTrigger render={<InputGroupButton size="xs" aria-label={label} />}>
        {icon}
        {label}
        <CountBadge count={value.length} />
      </ComboboxTrigger>
      <ComboboxContent side="top" className="w-72">
        <ComboboxInput showTrigger={false} placeholder={`Search ${label.toLowerCase()}`} />
        <ComboboxEmpty>
          {isLoading ? <Spinner className="size-3.5" /> : items.length === 0 && emptyText}
          {!isLoading && items.length > 0 && "Nothing matches"}
        </ComboboxEmpty>
        <GroupedOptions index={index} />
      </ComboboxContent>
    </Combobox>
  );
}

export function TargetPicker({
  label,
  placeholder,
  groups,
  value,
  onValueChange,
  isLoading,
  emptyText,
  allowCustom = false,
  defaultOpen = false,
}: {
  label: string;
  placeholder: string;
  groups: readonly OptionGroup[];
  value: string | null;
  onValueChange: (value: string) => void;
  isLoading: boolean;
  emptyText: string;
  allowCustom?: boolean;
  defaultOpen?: boolean;
}) {
  const [query, setQuery] = useState("");
  const index = optionIndex(groups);
  const custom = query.trim();
  const offerCustom = allowCustom && custom !== "" && !isListed(index, custom);
  const items = [...toItems(groups), ...(offerCustom ? [{ value: "Custom name", items: [custom] }] : [])];
  const unsetText = isLoading ? `Loading ${label.toLowerCase()}s` : placeholder;
  return (
    <Combobox
      items={items}
      value={value}
      onValueChange={(next: string | null) => next && onValueChange(next)}
      inputValue={query}
      onInputValueChange={setQuery}
      onOpenChange={(open) => !open && setQuery("")}
      itemToStringLabel={(item: string) => index.get(item)?.label ?? item}
      defaultOpen={defaultOpen}
      autoHighlight
    >
      <ComboboxTrigger render={<InputGroupButton size="xs" className="max-w-72 min-w-0" aria-label={label} />}>
        {isLoading && !value ? <Spinner className="size-3" /> : null}
        <span
          className={cn(
            "min-w-0 truncate font-mono text-xs font-medium",
            value ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {value ? index.get(value)?.label ?? value : unsetText}
        </span>
      </ComboboxTrigger>
      <ComboboxContent side="top" className="w-80">
        <ComboboxInput showTrigger={false} placeholder={`Search ${label.toLowerCase()}s`} />
        <ComboboxEmpty>
          {items.length === 0 && !custom ? emptyText : `No matching ${label.toLowerCase()}`}
        </ComboboxEmpty>
        <GroupedOptions index={index} mono />
      </ComboboxContent>
    </Combobox>
  );
}
