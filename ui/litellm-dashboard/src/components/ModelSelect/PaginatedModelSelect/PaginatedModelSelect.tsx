import { useInfiniteModelInfo } from "@/app/(dashboard)/hooks/models/useModels";
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

export interface PaginatedModelSelectProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  style?: React.CSSProperties;
  pageSize?: number;
  allowClear?: boolean;
  disabled?: boolean;
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

export const PaginatedModelSelect = ({
  value,
  onChange,
  placeholder = "Select a model",
  style,
  pageSize = 50,
  allowClear = true,
  disabled = false,
}: PaginatedModelSelectProps) => {
  const [open, setOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useDebouncedState("", {
    wait: DEBOUNCE_MS,
  });
  const listRef = useRef<HTMLDivElement>(null);

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useInfiniteModelInfo(pageSize, debouncedSearch || undefined);

  const options = useMemo(() => {
    if (!data?.pages) return [];

    const seen = new Set<string>();
    const result: {
      label: string;
      value: string;
      modelName: string;
      modelId: string;
    }[] = [];

    for (const page of data.pages) {
      for (const model of page.data) {
        const modelId = model.model_info?.id ?? "";
        const modelName = model.model_name ?? "";

        if (!modelId || seen.has(modelId)) continue;
        seen.add(modelId);

        result.push({
          label: modelName ? `${modelName} (${modelId})` : modelId,
          value: modelId,
          modelName,
          modelId,
        });
      }
    }

    return result;
  }, [data]);

  const selected = value ? options.find((o) => o.value === value) : undefined;

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

  const select = (id: string) => {
    onChange?.(id);
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
          {selected ? (
            <span className="truncate">{selected.label}</span>
          ) : value ? (
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
            placeholder="Search models…"
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
              No models found
            </div>
          ) : (
            options.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => select(option.value)}
                className={cn(
                  "w-full text-left px-2 py-2 text-sm rounded hover:bg-accent",
                  value === option.value && "bg-accent",
                )}
              >
                {option.modelName ? (
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <span className="font-bold">Model name:</span>
                      <span className="truncate">{option.modelName}</span>
                    </div>
                    <span className="truncate text-muted-foreground">
                      Model ID: {option.modelId}
                    </span>
                  </div>
                ) : (
                  <span className="truncate text-muted-foreground">
                    Model ID: {option.modelId}
                  </span>
                )}
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
