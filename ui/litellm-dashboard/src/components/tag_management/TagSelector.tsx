import React, { useEffect, useMemo, useRef, useState } from "react";
import { Tag } from "./types";
import { tagListCall } from "../networking";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface TagSelectorProps {
  onChange: (selectedTags: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
}

/**
 * Free-form tag input. Supports:
 *   - selecting from a server-fetched suggestion list (popover)
 *   - typing a new tag and pressing Enter / typing a `,` to add it
 *   - removing existing tags via the X icon on each chip
 *
 * Replaces antd `<Select mode="tags">` for the phase-1 migration.
 */
const TagSelector: React.FC<TagSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
}) => {
  const [tags, setTags] = useState<Tag[]>([]);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const selected = useMemo(() => value ?? [], [value]);

  useEffect(() => {
    const fetchTags = async () => {
      if (!accessToken) return;
      try {
        const response = await tagListCall(accessToken);
        setTags(Object.values(response));
      } catch (error) {
        console.error("Error fetching tags:", error);
      }
    };
    fetchTags();
  }, [accessToken]);

  const suggestions = useMemo(
    () =>
      tags
        .filter((t) => !selected.includes(t.name))
        .filter((t) =>
          query ? t.name.toLowerCase().includes(query.toLowerCase()) : true,
        )
        .slice(0, 10),
    [tags, selected, query],
  );

  const addTag = (name: string) => {
    const trimmed = name.trim();
    if (!trimmed || selected.includes(trimmed)) return;
    onChange([...selected, trimmed]);
    setQuery("");
  };

  const removeTag = (name: string) => {
    onChange(selected.filter((t) => t !== name));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (query.trim()) addTag(query);
    } else if (e.key === "," ) {
      e.preventDefault();
      if (query.trim()) addTag(query);
    } else if (
      e.key === "Backspace" &&
      query === "" &&
      selected.length > 0
    ) {
      onChange(selected.slice(0, -1));
    }
  };

  return (
    <Popover open={open && suggestions.length > 0} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <div
          className={cn(
            "min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm cursor-text",
            className,
          )}
          onClick={() => inputRef.current?.focus()}
        >
          {selected.map((name) => (
            <Badge key={name} variant="secondary" className="gap-1">
              {name}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  removeTag(name);
                }}
                className="inline-flex items-center"
                aria-label={`Remove ${name}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
          <Input
            ref={inputRef}
            value={query}
            placeholder={selected.length === 0 ? "Select or create tags" : ""}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={handleKeyDown}
            className="border-0 shadow-none focus-visible:ring-0 h-7 px-1 flex-1 min-w-[120px]"
          />
        </div>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] p-1"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        {suggestions.map((t) => (
          <button
            key={t.name}
            type="button"
            className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent"
            onClick={() => {
              addTag(t.name);
              inputRef.current?.focus();
            }}
            title={t.description || t.name}
          >
            {t.name}
          </button>
        ))}
      </PopoverContent>
    </Popover>
  );
};

export default TagSelector;
