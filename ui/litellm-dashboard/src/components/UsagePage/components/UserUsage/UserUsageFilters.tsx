/**
 * User Usage Filters Component
 *
 * Filter controls for tag selection, spend thresholds, and sorting
 */

import { Button, Card } from "@tremor/react";
import { Input, Select } from "antd";
import React, { useState } from "react";
import { UserUsageFiltersState } from "./types";

interface UserUsageFiltersProps {
  filters: UserUsageFiltersState;
  onFiltersChange: (filters: Partial<UserUsageFiltersState>) => void;
  loading: boolean;
}

const COMMON_TAGS = [
  "User-Agent:claude-code",
  "User-Agent:claude-code-max",
  "User-Agent:cursor",
  "User-Agent:windsurf",
];

export const UserUsageFilters: React.FC<UserUsageFiltersProps> = ({
  filters,
  onFiltersChange,
  loading,
}) => {
  const [localMinSpend, setLocalMinSpend] = useState<string>(
    filters.minSpend?.toString() || ""
  );
  const [localMaxSpend, setLocalMaxSpend] = useState<string>(
    filters.maxSpend?.toString() || ""
  );

  const handleApplySpendFilters = () => {
    onFiltersChange({
      minSpend: localMinSpend ? parseFloat(localMinSpend) : null,
      maxSpend: localMaxSpend ? parseFloat(localMaxSpend) : null,
    });
  };

  const handleResetFilters = () => {
    setLocalMinSpend("");
    setLocalMaxSpend("");
    onFiltersChange({
      tagFilters: [],
      minSpend: null,
      maxSpend: null,
      sortBy: "spend",
      sortOrder: "desc",
    });
  };

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Filters</h3>
          <Button
            size="xs"
            variant="secondary"
            onClick={handleResetFilters}
            disabled={loading}
          >
            Reset All
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Tag Filters */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              🏷️ User Agent Tags
            </label>
            <Select
              mode="multiple"
              style={{ width: "100%" }}
              placeholder="Select tags..."
              value={filters.tagFilters}
              onChange={(value) => onFiltersChange({ tagFilters: value })}
              disabled={loading}
              options={COMMON_TAGS.map((tag) => ({
                label: tag,
                value: tag,
              }))}
              allowClear
            />
          </div>

          {/* Min Spend */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              💰 Min Spend ($)
            </label>
            <Input
              type="number"
              placeholder="e.g., 200"
              value={localMinSpend}
              onChange={(e) => setLocalMinSpend(e.target.value)}
              onPressEnter={handleApplySpendFilters}
              disabled={loading}
              min={0}
              step={10}
            />
          </div>

          {/* Max Spend */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              💰 Max Spend ($)
            </label>
            <Input
              type="number"
              placeholder="e.g., 1000"
              value={localMaxSpend}
              onChange={(e) => setLocalMaxSpend(e.target.value)}
              onPressEnter={handleApplySpendFilters}
              disabled={loading}
              min={0}
              step={10}
            />
          </div>

          {/* Sort By */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              📊 Sort By
            </label>
            <div className="flex gap-2">
              <Select
                style={{ flex: 1 }}
                value={filters.sortBy}
                onChange={(value) => onFiltersChange({ sortBy: value })}
                disabled={loading}
                options={[
                  { label: "Spend", value: "spend" },
                  { label: "Requests", value: "requests" },
                  { label: "Tokens", value: "tokens" },
                ]}
              />
              <Select
                style={{ width: 100 }}
                value={filters.sortOrder}
                onChange={(value) => onFiltersChange({ sortOrder: value })}
                disabled={loading}
                options={[
                  { label: "↓ Desc", value: "desc" },
                  { label: "↑ Asc", value: "asc" },
                ]}
              />
            </div>
          </div>
        </div>

        {/* Apply Button for Spend Filters */}
        {(localMinSpend !== (filters.minSpend?.toString() || "") ||
          localMaxSpend !== (filters.maxSpend?.toString() || "")) && (
          <div className="flex justify-end">
            <Button onClick={handleApplySpendFilters} disabled={loading}>
              Apply Spend Filters
            </Button>
          </div>
        )}

        {/* Active Filters Summary */}
        {(filters.tagFilters.length > 0 ||
          filters.minSpend !== null ||
          filters.maxSpend !== null) && (
          <div className="text-sm text-gray-600">
            <strong>Active filters:</strong>{" "}
            {filters.tagFilters.length > 0 && (
              <span>Tags: {filters.tagFilters.length} selected; </span>
            )}
            {filters.minSpend !== null && (
              <span>Min spend: ${filters.minSpend}; </span>
            )}
            {filters.maxSpend !== null && (
              <span>Max spend: ${filters.maxSpend}; </span>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};
