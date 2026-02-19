import React from "react";
import { Button, TextInput } from "@tremor/react";
import { Filter, RotateCcw, Search } from "lucide-react";
import { Select } from "antd";

interface CustomersFiltersProps {
  filters: {
    user_id: string;
    alias: string;
    blocked: string;
    region: string;
  };
  showFilters: boolean;
  onToggleFilters: (show: boolean) => void;
  onChange: (key: string, value: string) => void;
  onReset: () => void;
}

const CustomersFilters: React.FC<CustomersFiltersProps> = ({
  filters,
  showFilters,
  onToggleFilters,
  onChange,
  onReset,
}) => {
  return (
    <div className="flex flex-col space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <TextInput
            placeholder="Search by Customer ID or Name..."
            value={filters.user_id || filters.alias}
            onChange={(e) => onChange("user_id", e.target.value)}
            className="pl-10"
          />
        </div>
        <Button
          variant="secondary"
          onClick={() => onToggleFilters(!showFilters)}
          icon={Filter}
        >
          Filters
        </Button>
        <Button
          variant="secondary"
          onClick={onReset}
          icon={RotateCcw}
        >
          Reset
        </Button>
      </div>

      {showFilters && (
        <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-md border border-gray-200">
          <div className="flex items-center gap-2">
            <label className="text-xs font-medium text-gray-600">Status:</label>
            <Select
              value={filters.blocked || "all"}
              onChange={(value) => onChange("blocked", value)}
              style={{ width: 120 }}
              size="small"
            >
              <Select.Option value="">All</Select.Option>
              <Select.Option value="active">Active</Select.Option>
              <Select.Option value="blocked">Blocked</Select.Option>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs font-medium text-gray-600">Region:</label>
            <Select
              value={filters.region || "all"}
              onChange={(value) => onChange("region", value)}
              style={{ width: 120 }}
              size="small"
            >
              <Select.Option value="">All</Select.Option>
              <Select.Option value="us">US</Select.Option>
              <Select.Option value="eu">EU</Select.Option>
            </Select>
          </div>
        </div>
      )}
    </div>
  );
};

export default CustomersFilters;
