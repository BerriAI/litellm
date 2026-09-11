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
import { getClaudeCodePluginsList } from "../networking";

export interface SkillSelectorProps {
  onChange: (selected: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

interface SkillOption {
  readonly name: string;
  readonly enabled: boolean;
}

const readSkillOptions = (data: unknown): SkillOption[] => {
  const plugins = (data as { plugins?: unknown[] } | undefined)?.plugins;
  if (!Array.isArray(plugins)) return [];
  return plugins.flatMap((plugin) => {
    const record = plugin as { name?: unknown; enabled?: unknown };
    return typeof record.name === "string" && record.name.length > 0
      ? [{ name: record.name, enabled: record.enabled !== false }]
      : [];
  });
};

const SkillSelector: React.FC<SkillSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select skills (optional)",
  disabled = false,
}) => {
  const anchor = useComboboxAnchor();
  const [options, setOptions] = useState<SkillOption[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        setOptions(readSkillOptions(await getClaudeCodePluginsList(accessToken)));
      } catch (e) {
        console.error("Failed to load skills:", e);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [accessToken]);

  return (
    <Combobox
      multiple
      items={options.map((option) => option.name)}
      value={value ?? []}
      onValueChange={(selected: string[]) => onChange(selected)}
      disabled={disabled}
    >
      <ComboboxChips render={<div ref={anchor} />} className={cn("w-full", className)} aria-busy={loading}>
        <ComboboxValue>
          {(selected: string[]) =>
            selected.map((skill) => (
              <ComboboxChip key={skill} aria-label={skill}>
                {skill}
              </ComboboxChip>
            ))
          }
        </ComboboxValue>
        <ComboboxChipsInput placeholder={placeholder} aria-label={placeholder} disabled={disabled} />
        {value && value.length > 0 && <ComboboxClear aria-label="Clear all skills" disabled={disabled} />}
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{loading ? "Loading skills…" : "No skills found"}</ComboboxEmpty>
        <ComboboxList>
          {(skill: string) => {
            const isPrivate = options.some((option) => option.name === skill && !option.enabled);
            return (
              <ComboboxItem key={skill} value={skill} aria-label={isPrivate ? `${skill} (private)` : skill}>
                {skill}
                {isPrivate && <span className="ml-2 text-xs text-muted-foreground">private</span>}
              </ComboboxItem>
            );
          }}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
};

export default SkillSelector;
