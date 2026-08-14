import type { DateRangePickerValue } from "@tremor/react";
import { Download } from "lucide-react";
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxClear,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
} from "@/components/ui/combobox";
import EntityUsageExportModal from "./EntityUsageExportModal";
import type { EntitySpendData, EntityType } from "./types";
import type { Team } from "@/components/key_team_helpers/key_list";

interface UsageExportHeaderProps {
  dateValue: DateRangePickerValue;
  entityType: EntityType;
  spendData: EntitySpendData;
  // Optional filter props
  showFilters?: boolean;
  filterLabel?: string;
  filterPlaceholder?: string;
  selectedFilters?: string[];
  onFiltersChange?: (filters: string[]) => void;
  filterOptions?: Array<{ label: string; value: string }>;
  filterMode?: "multiple" | "single";
  customTitle?: string;
  compactLayout?: boolean;
  teams?: Team[];
}

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
  compactLayout = false,
  teams = [],
}) => {
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  const hasFilters = showFilters && filterOptions.length > 0;
  const optionValues = filterOptions.map((option) => option.value);
  const labelOf = (value: string) => filterOptions.find((option) => option.value === value)?.label ?? value;

  const filterList = (
    <ComboboxContent>
      <ComboboxEmpty>No options found</ComboboxEmpty>
      <ComboboxList>
        {(value: string) => (
          <ComboboxItem key={value} value={value}>
            {labelOf(value)}
          </ComboboxItem>
        )}
      </ComboboxList>
    </ComboboxContent>
  );

  return (
    <>
      <div className="mb-4">
        {/**
         * Use CSS grid with items-end so all cells (filter, button)
         * align to the same baseline regardless of label heights. This removes
         * vertical drift when the right column has a label above the input.
         */}
        <div className={`grid ${hasFilters ? "grid-cols-[1fr_auto]" : "grid-cols-[auto]"} items-end gap-4`}>
          {hasFilters && (
            <div>
              {filterLabel && <label className="text-sm font-medium text-gray-700 block mb-2">{filterLabel}</label>}
              {filterMode === "single" ? (
                <Combobox
                  items={optionValues}
                  value={selectedFilters[0] ?? null}
                  onValueChange={(next: string | null) => onFiltersChange?.(next ? [next] : [])}
                  itemToStringLabel={labelOf}
                >
                  <ComboboxInput
                    className="w-full"
                    placeholder={filterPlaceholder}
                    aria-label={filterPlaceholder}
                    showClear={selectedFilters.length > 0}
                  />
                  {filterList}
                </Combobox>
              ) : (
                <Combobox
                  multiple
                  items={optionValues}
                  value={selectedFilters}
                  onValueChange={(next: string[]) => onFiltersChange?.(next)}
                >
                  <ComboboxChips className="w-full">
                    <ComboboxValue>
                      {(selected: string[]) =>
                        selected.map((value) => (
                          <ComboboxChip key={value} aria-label={labelOf(value)}>
                            {labelOf(value)}
                          </ComboboxChip>
                        ))
                      }
                    </ComboboxValue>
                    <ComboboxChipsInput
                      className="border-0 bg-transparent"
                      placeholder={filterPlaceholder}
                      aria-label={filterPlaceholder}
                    />
                    {selectedFilters.length > 0 && <ComboboxClear aria-label={`Clear ${filterLabel ?? "filters"}`} />}
                  </ComboboxChips>
                  {filterList}
                </Combobox>
              )}
            </div>
          )}

          <div className="justify-self-end">
            <Button onClick={() => setIsExportModalOpen(true)}>
              <Download />
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
