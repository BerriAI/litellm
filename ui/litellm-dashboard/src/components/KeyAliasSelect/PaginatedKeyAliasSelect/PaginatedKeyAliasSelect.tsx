import { useInfiniteKeyAliases } from "@/app/(dashboard)/hooks/keys/useKeyAliases";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { ChevronsUpDown, Loader2, X } from "lucide-react";
import { useMemo, useRef, useState } from "react";

export interface PaginatedKeyAliasSelectProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  style?: React.CSSProperties;
  pageSize?: number;
  allowClear?: boolean;
  disabled?: boolean;
  allFilters?: { [key: string]: string };
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

export const PaginatedKeyAliasSelect = ({
  value,
  onChange,
  placeholder = "Select a key alias",
  style,
  pageSize = 50,
  allowClear = true,
  disabled = false,
  allFilters,
}: PaginatedKeyAliasSelectProps) => {
  const [open, setOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useDebouncedState("", {
    wait: DEBOUNCE_MS,
  });
  const listRef = useRef<HTMLDivElement>(null);

  const teamId = allFilters?.["Team ID"] || undefined;

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useInfiniteKeyAliases(pageSize, debouncedSearch || undefined, teamId);

  const options = useMemo(() => {
    if (!data?.pages) return [];

    const seen = new Set<string>();
    const result: string[] = [];

    for (const page of data.pages) {
      for (const alias of page.aliases) {
        if (!alias || seen.has(alias)) continue;
        seen.add(alias);
        result.push(alias);
      }
    }

    return result;
  }, [data]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const scrollRatio =
      (target.scrollTop + target.clientHeight) / target.scrollHeight;

    if (scrollRatio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  };

  const handleSearch = (v: string) => {
    setSearchInput(v);
    setDebouncedSearch(v);
  };

  const select = (alias: string) => {
    onChange?.(alias);
    setOpen(false);
  };

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onChange?.("");
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          style={style}
          className="w-full justify-between"
        >
          {value ? (
            <span className="truncate">{value}</span>
          ) : (
            <span className="text-muted-foreground">{placeholder}</span>
          )}
          <div className="ml-2 flex items-center gap-1 shrink-0">
            {allowClear && value && (
              <span
                role="button"
                tabIndex={0}
                onClick={clear}
                className="text-muted-foreground hover:text-foreground"
                aria-label="Clear"
              >
                <X className="h-3.5 w-3.5" />
              </span>
            )}
            <ChevronsUpDown className="h-4 w-4 opacity-50" />
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[var(--radix-popover-trigger-width)] p-0"
        align="start"
      >
        <div className="p-2 border-b">
          <Input
            autoFocus
            placeholder="Search key aliases…"
            value={searchInput}
            onChange={(e) => handleSearch(e.target.value)}
            className="h-8"
          />
        </div>
        <div
          ref={listRef}
          onScroll={handleScroll}
          className="max-h-60 overflow-y-auto p-1"
        >
          {isLoading && options.length === 0 ? (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          ) : options.length === 0 ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              No key aliases found
            </div>
          ) : (
            options.map((alias) => (
              <button
                key={alias}
                type="button"
                onClick={() => select(alias)}
                className={cn(
                  "w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent",
                  value === alias && "bg-accent",
                )}
              >
                {alias}
              </button>
            ))
          )}
          {isFetchingNextPage && (
            <div className="flex items-center justify-center py-2">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};
