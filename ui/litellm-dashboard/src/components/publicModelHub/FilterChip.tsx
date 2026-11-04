"use client";

import { CirclePlus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
  ComboboxTrigger,
} from "@/components/ui/combobox";
import { cn } from "@/lib/cva.config";

import { firstPlus } from "./hubFormat";
import { ProviderAvatar } from "./HubParts";

interface FilterOption {
  value: string;
  label: string;
  provider?: string;
}

export function FilterChip({
  label,
  noun,
  options,
  value,
  onChange,
}: {
  label: string;
  noun: string;
  options: FilterOption[];
  value: string[];
  onChange: (value: string[]) => void;
}) {
  const byValue = new Map(options.map((option) => [option.value, option]));
  const selected = value.map((v) => byValue.get(v)?.label ?? v);
  const active = value.length > 0;
  return (
    <ButtonGroup>
      <Combobox<FilterOption, true>
        multiple
        items={options}
        value={options.filter((option) => value.includes(option.value))}
        onValueChange={(next) => onChange(next.map((option) => option.value))}
        itemToStringLabel={(option) => option.label}
        isItemEqualToValue={(a, b) => a.value === b.value}
      >
        <ComboboxTrigger
          render={
            <Button
              variant="outline"
              size="sm"
              className={cn("[&>svg:last-child]:hidden", !active && "border-dashed text-muted-foreground")}
            />
          }
        >
          {!active && <CirclePlus />}
          {label}
          {active && <span className="font-normal text-muted-foreground">{firstPlus(selected)}</span>}
        </ComboboxTrigger>
        <ComboboxContent className="w-auto min-w-60">
          <ComboboxInput showTrigger={false} placeholder={`Search ${noun}`} />
          <ComboboxEmpty>No {noun} found</ComboboxEmpty>
          <ComboboxList>
            {(option: FilterOption) => (
              <ComboboxItem key={option.value} value={option}>
                {option.provider && <ProviderAvatar provider={option.provider} />}
                <span className="whitespace-nowrap">{option.label}</span>
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
      {active && (
        <Button
          variant="outline"
          size="icon-sm"
          aria-label={`Clear ${label.toLowerCase()} filter`}
          onClick={() => onChange([])}
        >
          <X />
        </Button>
      )}
    </ButtonGroup>
  );
}
