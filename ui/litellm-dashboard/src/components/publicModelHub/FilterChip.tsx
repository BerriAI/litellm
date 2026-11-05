"use client";

import { CirclePlus, X } from "lucide-react";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
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
          <ComboboxInput showTrigger={false} placeholder={t("publicModelHub.filterChip.search", { noun })} />
          <ComboboxEmpty>{t("publicModelHub.filterChip.empty", { noun })}</ComboboxEmpty>
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
          aria-label={t("publicModelHub.filterChip.clear", { label: label.toLowerCase() })}
          onClick={() => onChange([])}
        >
          <X />
        </Button>
      )}
    </ButtonGroup>
  );
}
