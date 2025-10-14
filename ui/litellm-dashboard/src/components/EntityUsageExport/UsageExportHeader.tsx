import React, { useState } from "react";
import { Button, Grid, Col, Text } from "@tremor/react";
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
}) => {
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  return (
    <>
      <div className="mb-4">
        <div className="flex justify-between items-end gap-4">
          <Grid numItems={2} className="gap-2 flex-1">
            <Col>
              <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
            </Col>
            {showFilters && filterOptions.length > 0 && (
              <Col>
                {filterLabel && <Text>{filterLabel}</Text>}
                <Select
                  mode="multiple"
                  style={{ width: "100%" }}
                  placeholder={filterPlaceholder}
                  value={selectedFilters}
                  onChange={onFiltersChange}
                  options={filterOptions}
                  className="mt-2"
                  allowClear
                />
              </Col>
            )}
          </Grid>
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

      <EntityUsageExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        entityType={entityType}
        spendData={spendData}
        dateRange={dateValue}
        selectedFilters={selectedFilters}
      />
    </>
  );
};

export default UsageExportHeader;

