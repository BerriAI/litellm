/**
 * Unified selector component that handles both model and agent selection
 * based on the current endpoint configuration.
 */

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { SelectorOption, EndpointConfig } from "../endpoint_config";

interface UnifiedSelectorProps {
  value: string;
  options: SelectorOption[];
  loading: boolean;
  config: EndpointConfig;
  onChange: (value: string) => void;
}

const matchesQuery = (option: SelectorOption, query: string): boolean =>
  option.label.toLowerCase().includes(query.trim().toLowerCase());

export function UnifiedSelector({ value, options, loading, config, onChange }: UnifiedSelectorProps) {
  const selected = options.find((option) => option.value === value) ?? null;
  const noun = config.selectorLabel.toLowerCase();

  return (
    <Combobox
      items={options}
      value={selected}
      onValueChange={(option: SelectorOption | null) => onChange(option?.value ?? "")}
      isItemEqualToValue={(a: SelectorOption, b: SelectorOption) => a.value === b.value}
      itemToStringLabel={(option: SelectorOption) => option.label}
      filter={matchesQuery}
    >
      <ComboboxInput
        placeholder={loading ? `Loading ${noun}s...` : config.selectorPlaceholder}
        className="w-48 md:w-64 lg:w-72"
      />
      <ComboboxContent>
        <ComboboxEmpty>
          {loading ? (
            <span aria-busy="true" className="flex items-center justify-center py-2">
              <UiLoadingSpinner className="size-4" />
            </span>
          ) : (
            `No ${noun}s available`
          )}
        </ComboboxEmpty>
        <ComboboxList>
          {(option: SelectorOption) => (
            <ComboboxItem key={option.value} value={option}>
              {option.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}
