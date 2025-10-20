import React, { useState } from "react";
import { Button, Text } from "@tremor/react";
import { Select } from "antd";
import AdvancedDatePicker from "../shared/advanced_date_picker";
import EntityUsageExportModal from "./EntityUsageExportModal";
import type { DateRangePickerValue } from "@tremor/react";
import type { EntitySpendData } from "./types";

interface UsageExportHeaderProps {
  dateValue: DateRangePickerValue;
  onDateChange: (value: DateRangePickerValue) => void;
  entityType: "tag" | "team";
  spendData: EntitySpendData;
  // Optional filter props
  showFilters?: boolean;
  filterLabel?: string;
  filterPlaceholder?: string;
  selectedFilters?: string[];
  onFiltersChange?: (filters: string[]) => void;
  filterOptions?: Array<{ label: string; value: string }>;
  customTitle?: string;
  compactLayout?: boolean;
}

const UsageExportHeader: React.FC<UsageExportHeaderProps> = ({
  dateValue,
  onDateChange,
  entityType,
  spendData,
  showFilters = false,
  filterLabel,
  filterPlaceholder,
  selectedFilters = [],
  onFiltersChange,
  filterOptions = [],
  customTitle,
  compactLayout = false,
}) => {
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  return (
    <>
      <div className="mb-4">
        {/**
         * Use CSS grid with items-end so all cells (date picker, filter, button)
         * align to the same baseline regardless of label heights. This removes
         * vertical drift when the right column has a label above the input.
         */}
        <div
          className={`grid ${
            showFilters && filterOptions.length > 0 ? "grid-cols-[1fr_1fr_auto]" : "grid-cols-[1fr_auto]"
          } items-end gap-4`}
        >
          <div>
            <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
          </div>

          {showFilters && filterOptions.length > 0 && (
            <div>
              {filterLabel && <Text className="mb-2">{filterLabel}</Text>}
              <Select
                mode="multiple"
                style={{ width: "100%" }}
                placeholder={filterPlaceholder}
                value={selectedFilters}
                onChange={onFiltersChange}
                options={filterOptions}
                allowClear
              />
            </div>
          )}

          <div className="justify-self-end">
            <Button
              onClick={() => setIsExportModalOpen(true)}
              icon={() => (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
              )}
            >
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
      />
    </>
  );
};

export default UsageExportHeader;

