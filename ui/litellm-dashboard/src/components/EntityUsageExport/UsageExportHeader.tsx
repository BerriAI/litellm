// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import type { DateRangePickerValue } from "@tremor/react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Download, X } from "lucide-react";
import React, { useState } from "react";
import EntityUsageExportModal from "./EntityUsageExportModal";
import type { EntitySpendData, EntityType } from "./types";
import type { Team } from "@/components/key_team_helpers/key_list";
import { cn } from "@/lib/utils";

interface UsageExportHeaderProps {
  dateValue: DateRangePickerValue;
  entityType: EntityType;
  spendData: EntitySpendData;
  showFilters?: boolean;
  filterLabel?: string;
  filterPlaceholder?: string;
  selectedFilters?: string[];
  onFiltersChange?: (filters: string[]) => void;
  filterOptions?: Array<{ label: string; value: string }>;
  filterMode?: "multiple" | "single";
  customTitle?: string;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  compactLayout?: boolean;
  teams?: Team[];
}

const ALL = "__all__";

const UsageExportHeader: React.FC<UsageExportHeaderProps> = ({
  dateValue,
  entityType,
  spendData,
  showFilters = false,
  filterLabel,
  filterPlaceholder,
  selectedFilters = [],
  onFiltersChange,
  filterOptions = [],
  filterMode = "multiple",
  customTitle,
  teams = [],
}) => {
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [multiOpen, setMultiOpen] = useState(false);

  const hasFilters = showFilters && filterOptions.length > 0;
  const gridCols = hasFilters ? "grid-cols-[1fr_auto]" : "grid-cols-[auto]";

  const labelFor = (v: string) =>
    filterOptions.find((o) => o.value === v)?.label ?? v;

  return (
    <>
      <div className="mb-4">
        <div className={`grid ${gridCols} items-end gap-4`}>
          {hasFilters && (
            <div>
              {filterLabel && (
                <p className="mb-2 text-sm text-muted-foreground">
                  {filterLabel}
                </p>
              )}
              {filterMode === "single" ? (
                <Select
                  value={selectedFilters[0] ?? ALL}
                  onValueChange={(v) =>
                    onFiltersChange?.(v === ALL ? [] : [v])
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder={filterPlaceholder} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={ALL}>All</SelectItem>
                    {filterOptions.map((o) => (
                      <SelectItem key={o.value} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <Popover open={multiOpen} onOpenChange={setMultiOpen}>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      className={cn(
                        "min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left",
                      )}
                    >
                      {selectedFilters.length === 0 ? (
                        <span className="text-muted-foreground px-1">
                          {filterPlaceholder ?? "Select…"}
                        </span>
                      ) : (
                        selectedFilters.map((v) => (
                          <Badge
                            key={v}
                            variant="secondary"
                            className="gap-1 inline-flex items-center"
                          >
                            {labelFor(v)}
                            <span
                              role="button"
                              tabIndex={0}
                              onClick={(e) => {
                                e.stopPropagation();
                                onFiltersChange?.(
                                  selectedFilters.filter((s) => s !== v),
                                );
                              }}
                              className="inline-flex items-center"
                              aria-label={`Remove ${labelFor(v)}`}
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
                    className="w-[var(--radix-popover-trigger-width)] p-1 max-h-60 overflow-y-auto"
                  >
                    {filterOptions
                      .filter((o) => !selectedFilters.includes(o.value))
                      .map((opt) => (
                        <button
                          key={opt.value}
                          type="button"
                          className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent"
                          onClick={() =>
                            onFiltersChange?.([...selectedFilters, opt.value])
                          }
                        >
                          {opt.label}
                        </button>
                      ))}
                    {filterOptions.every((o) =>
                      selectedFilters.includes(o.value),
                    ) && (
                      <div className="py-2 px-3 text-sm text-muted-foreground">
                        All selected
                      </div>
                    )}
                  </PopoverContent>
                </Popover>
              )}
            </div>
          )}

          <div className="justify-self-end">
            <Button onClick={() => setIsExportModalOpen(true)}>
              <Download className="h-4 w-4" />
              Export Data
            </Button>
          </div>
        </div>
      </div>

      <EntityUsageExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        entityType={entityType}
        spendData={spendData}
        dateRange={dateValue}
        selectedFilters={selectedFilters}
        customTitle={customTitle}
        teams={teams}
      />
    </>
  );
};

export default UsageExportHeader;
