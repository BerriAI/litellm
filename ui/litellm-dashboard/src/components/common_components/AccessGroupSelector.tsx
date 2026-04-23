import React, { useMemo, useState } from "react";
import { Users, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import {
  useAccessGroups,
  AccessGroupResponse,
} from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

export interface AccessGroupSelectorProps {
  value?: string[];
  onChange?: (value: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  style?: React.CSSProperties;
  className?: string;
  showLabel?: boolean;
  labelText?: string;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  allowClear?: boolean;
}

/**
 * Reusable multi-select for access groups (shadcn Popover + chip list).
 *
 * - Displays the `access_group_name` in chips and options.
 * - Returns an array of `access_group_id` values.
 * - Drop-in replacement for the old antd Select mode='multiple'.
 */
const AccessGroupSelector: React.FC<AccessGroupSelectorProps> = ({
  value,
  onChange,
  placeholder = "Select access groups",
  disabled = false,
  style,
  className,
  showLabel = false,
  labelText = "Access Group",
}) => {
  const { data: accessGroups, isLoading, isError } = useAccessGroups();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const selected = value ?? [];

  const allOptions = useMemo(
    () =>
      (accessGroups ?? []).map((group: AccessGroupResponse) => ({
        id: group.access_group_id,
        name: group.access_group_name,
        searchText: `${group.access_group_name} ${group.access_group_id}`,
      })),
    [accessGroups],
  );

  const filteredOptions = useMemo(
    () =>
      allOptions
        .filter((o) => !selected.includes(o.id))
        .filter((o) =>
          query
            ? o.searchText.toLowerCase().includes(query.toLowerCase())
            : true,
        ),
    [allOptions, selected, query],
  );

  const labelFor = (id: string) =>
    allOptions.find((o) => o.id === id)?.name ?? id;

  if (isLoading) {
    return (
      <div>
        {showLabel && (
          <div className="font-medium mb-2 text-foreground flex items-center">
            <Users className="mr-2 h-4 w-4" /> {labelText}
          </div>
        )}
        <Skeleton className="h-8 w-full" style={style} />
      </div>
    );
  }

  return (
    <div>
      {showLabel && (
        <div className="font-medium mb-2 text-foreground flex items-center">
          <Users className="mr-2 h-4 w-4" /> {labelText}
        </div>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            disabled={disabled}
            style={style}
            className={cn(
              "min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left disabled:opacity-50",
              className,
            )}
          >
            {selected.length === 0 ? (
              <span className="text-muted-foreground px-1">{placeholder}</span>
            ) : (
              selected.map((id) => (
                <Badge
                  key={id}
                  variant="secondary"
                  className="gap-1 inline-flex items-center"
                >
                  {labelFor(id)}
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      onChange?.(selected.filter((s) => s !== id));
                    }}
                    className="inline-flex items-center"
                    aria-label={`Remove ${labelFor(id)}`}
                  >
                    <X size={12} />
                  </span>
                </Badge>
              ))
            )}
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="w-[var(--radix-popover-trigger-width)] p-2"
        >
          <Input
            autoFocus
            placeholder="Search access groups…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-8 mb-2"
          />
          <div className="max-h-60 overflow-y-auto">
            {isError ? (
              <div className="py-2 px-3 text-sm text-destructive">
                Failed to load access groups
              </div>
            ) : filteredOptions.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No access groups found
              </div>
            ) : (
              filteredOptions.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent flex items-center gap-2"
                  onClick={() => onChange?.([...selected, opt.id])}
                >
                  <span className="font-medium">{opt.name}</span>
                  <span className="text-muted-foreground text-xs">
                    ({opt.id})
                  </span>
                </button>
              ))
            )}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
};

export default AccessGroupSelector;
