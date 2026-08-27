import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
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
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
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
  filterSlot?: React.ReactNode;
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
  filterSlot,
  customTitle,
  compactLayout = false,
  teams = [],
}) => {
  const anchor = useComboboxAnchor();
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  const hasFilters = filterSlot != null || (showFilters && filterOptions.length > 0);
  const optionValues = filterOptions.map((option) => option.value);
  const labelOf = (value: string) => filterOptions.find((option) => option.value === value)?.label ?? value;

  const filterList = (
    <ComboboxContent anchor={anchor}>
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

  const builtInFilter = (
    <Combobox
      multiple
      items={optionValues}
      value={selectedFilters}
      onValueChange={(next: string[]) => onFiltersChange?.(next)}
    >
      <ComboboxChips render={<div ref={anchor} />} className="w-full">
        <ComboboxValue>
          {(selected: string[]) =>
            selected.map((value) => (
              <ComboboxChip key={value} aria-label={labelOf(value)}>
                {labelOf(value)}
              </ComboboxChip>
            ))
          }
        </ComboboxValue>
        <ComboboxChipsInput placeholder={filterPlaceholder} aria-label={filterPlaceholder} />
        {selectedFilters.length > 0 && <ComboboxClear aria-label={`Clear ${filterLabel ?? "filters"}`} />}
      </ComboboxChips>
      {filterList}
    </Combobox>
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
              {filterLabel && <label className="text-sm font-medium text-foreground block mb-2">{filterLabel}</label>}
              {filterSlot ?? builtInFilter}
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
