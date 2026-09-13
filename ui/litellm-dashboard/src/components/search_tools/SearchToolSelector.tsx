import React, { useEffect, useState } from "react";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxClear,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { cn } from "@/lib/cva.config";
import { fetchSearchTools } from "../networking";

export interface SearchToolSelectorProps {
  onChange: (selected: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

const SearchToolSelector: React.FC<SearchToolSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select search tools (optional)",
  disabled = false,
}) => {
  const anchor = useComboboxAnchor();
  const [options, setOptions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const data = await fetchSearchTools(accessToken);
        const tools = Array.isArray(data?.search_tools)
          ? data.search_tools
          : Array.isArray(data?.data)
            ? data.data
            : [];
        setOptions(
          tools
            .map((tool: { search_tool_name?: string }) => tool?.search_tool_name)
            .filter((name: unknown): name is string => typeof name === "string" && name.length > 0),
        );
      } catch (e) {
        console.error("Failed to load search tools:", e);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [accessToken]);

  return (
    <Combobox
      multiple
      items={options}
      value={value ?? []}
      onValueChange={(selected: string[]) => onChange(selected)}
      disabled={disabled}
    >
      <ComboboxChips render={<div ref={anchor} />} className={cn("w-full", className)} aria-busy={loading}>
        <ComboboxValue>
          {(selected: string[]) =>
            selected.map((tool) => (
              <ComboboxChip key={tool} aria-label={tool}>
                {tool}
              </ComboboxChip>
            ))
          }
        </ComboboxValue>
        <ComboboxChipsInput placeholder={placeholder} aria-label={placeholder} disabled={disabled} />
        {value && value.length > 0 && <ComboboxClear aria-label="Clear all search tools" disabled={disabled} />}
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{loading ? "Loading search tools…" : "No search tools found"}</ComboboxEmpty>
        <ComboboxList>
          {(tool: string) => (
            <ComboboxItem key={tool} value={tool}>
              {tool}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
};

export default SearchToolSelector;
