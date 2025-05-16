import React, { useState, useCallback } from "react";
import { Button, Input, Select } from "antd";
import { FilterIcon } from "@heroicons/react/outline";
import debounce from "lodash/debounce";

export interface FilterOption {
  name: string;
  label?: string;
  isSearchable?: boolean;
  searchFn?: (
    searchText: string
  ) => Promise<Array<{ label: string; value: string }>>;
  options?: Array<{ label: string; value: string }>;
}

interface FilterValues {
  [key: string]: string;
}

interface FilterComponentProps {
  options: FilterOption[];
  onApplyFilters: (filters: FilterValues) => void;
  initialValues?: FilterValues;
  buttonLabel?: string;
  onResetFilters: () => void;
}

const FilterComponent: React.FC<FilterComponentProps> = ({
  options,
  onApplyFilters,
  onResetFilters,
  initialValues = {},
  buttonLabel = "Filters",
}) => {
  const [showFilters, setShowFilters] = useState<boolean>(false);
  const [tempValues, setTempValues] = useState<FilterValues>(initialValues);
  const [searchOptionsMap, setSearchOptionsMap] = useState<{
    [key: string]: Array<{ label: string; value: string }>;
  }>({});
  const [searchLoadingMap, setSearchLoadingMap] = useState<{
    [key: string]: boolean;
  }>({});
  const [searchInputValueMap, setSearchInputValueMap] = useState<{
    [key: string]: string;
  }>({});

  const debouncedSearch = useCallback(
    debounce(async (value: string, option: FilterOption) => {
      if (!option.isSearchable || !option.searchFn) return;

      setSearchLoadingMap((prev) => ({ ...prev, [option.name]: true }));
      try {
        const results = await option.searchFn(value);
        setSearchOptionsMap((prev) => ({ ...prev, [option.name]: results }));
      } catch (error) {
        console.error("Error searching:", error);
        setSearchOptionsMap((prev) => ({ ...prev, [option.name]: [] }));
      } finally {
        setSearchLoadingMap((prev) => ({ ...prev, [option.name]: false }));
      }
    }, 300),
    []
  );

  const handleFilterChange = (name: string, value: string) => {
    const newValues = {
      ...tempValues,
      [name]: value,
    };
    setTempValues(newValues);
    onApplyFilters(newValues);
  };

  const resetFilters = () => {
    const emptyValues: FilterValues = {};
    options.forEach((option) => {
      emptyValues[option.name] = "";
    });
    setTempValues(emptyValues);
    onResetFilters();
  };

  // Define the order of filters
  const orderedFilters = [
    "Team ID",
    "Status",
    "Organization ID",
    "Key Alias",
    "User ID",
    "Key Hash",
  ];

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-6">
        <Button
          icon={<FilterIcon className="h-4 w-4" />}
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2"
        >
          {buttonLabel}
        </Button>
        <Button onClick={resetFilters}>Reset Filters</Button>
      </div>

      {showFilters && (
        <div className="grid grid-cols-3 gap-x-6 gap-y-4 mb-6">
          {orderedFilters.map((filterName) => {
            const option = options.find(
              (opt) => opt.label === filterName || opt.name === filterName
            );
            if (!option) return null;

            return (
              <div key={option.name} className="flex flex-col gap-2">
                <label className="text-sm text-gray-600">
                  {option.label || option.name}
                </label>
                {option.isSearchable ? (
                  <Select
                    showSearch
                    className="w-full"
                    placeholder={`Search ${option.label || option.name}...`}
                    value={tempValues[option.name] || undefined}
                    onChange={(value) => handleFilterChange(option.name, value)}
                    onSearch={(value) => {
                      setSearchInputValueMap((prev) => ({
                        ...prev,
                        [option.name]: value,
                      }));
                      if (option.searchFn) {
                        debouncedSearch(value, option);
                      }
                    }}
                    filterOption={false}
                    loading={searchLoadingMap[option.name]}
                    options={searchOptionsMap[option.name] || []}
                    allowClear
                  />
                ) : option.options ? (
                  <Select
                    className="w-full"
                    placeholder={`Select ${option.label || option.name}...`}
                    value={tempValues[option.name] || undefined}
                    onChange={(value) => handleFilterChange(option.name, value)}
                    allowClear
                  >
                    {option.options.map((opt) => (
                      <Select.Option key={opt.value} value={opt.value}>
                        {opt.label}
                      </Select.Option>
                    ))}
                  </Select>
                ) : (
                  <Input
                    className="w-full"
                    placeholder={`Enter ${option.label || option.name}...`}
                    value={tempValues[option.name] || ""}
                    onChange={(e) =>
                      handleFilterChange(option.name, e.target.value)
                    }
                    allowClear
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default FilterComponent;
