import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { FilterIcon } from "@heroicons/react/outline";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { Button, Input, Select } from "antd";
import React, { useCallback, useEffect, useState } from "react";

export interface FilterOptionCustomComponentProps {
  value?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  allFilters?: { [key: string]: string };
}

export interface FilterOption {
  name: string;
  label?: string;
  isSearchable?: boolean;
  searchFn?: (searchText: string) => Promise<Array<{ label: string; value: string }>>;
  options?: Array<{ label: string; value: string }>;
  customComponent?: React.ComponentType<FilterOptionCustomComponentProps>;
  loading?: boolean;
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
  const [initialOptionsLoaded, setInitialOptionsLoaded] = useState<{
    [key: string]: boolean;
  }>({});

  const debouncedSearch = useDebouncedCallback(
    async (value: string, option: FilterOption) => {
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
    },
    { wait: DEBOUNCE_WAIT_MS },
  );

  // Load initial options for searchable filters
  const loadInitialOptions = useCallback(
    async (option: FilterOption) => {
      if (!option.isSearchable || !option.searchFn || option.loading || initialOptionsLoaded[option.name]) return;

      setSearchLoadingMap((prev) => ({ ...prev, [option.name]: true }));
      setInitialOptionsLoaded((prev) => ({ ...prev, [option.name]: true }));

      try {
        // Load initial options with empty search to get some default results
        const results = await option.searchFn("");
        setSearchOptionsMap((prev) => ({ ...prev, [option.name]: results }));
      } catch (error) {
        console.error("Error loading initial options:", error);
        setSearchOptionsMap((prev) => ({ ...prev, [option.name]: [] }));
      } finally {
        setSearchLoadingMap((prev) => ({ ...prev, [option.name]: false }));
      }
    },
    [initialOptionsLoaded],
  );

  // Load initial options when filters are shown
  useEffect(() => {
    if (showFilters) {
      options.forEach((option) => {
        if (option.isSearchable && !initialOptionsLoaded[option.name]) {
          loadInitialOptions(option);
        }
      });
    }
  }, [showFilters, options, loadInitialOptions, initialOptionsLoaded]);

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

  // Handle dropdown open to load initial options
  const handleDropdownVisibleChange = (open: boolean, option: FilterOption) => {
    if (open && option.isSearchable && !initialOptionsLoaded[option.name]) {
      loadInitialOptions(option);
    }
  };

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
          {options.map((option) => {
            const isOptionLoading = searchLoadingMap[option.name] || option.loading;
            return (
              <div key={option.name} className="flex flex-col gap-2">
                <label className="text-sm text-gray-600">{option.label || option.name}</label>
                {option.isSearchable ? (
                  <Select
                    showSearch
                    className="w-full"
                    placeholder={`Search ${option.label || option.name}...`}
                    value={tempValues[option.name] || undefined}
                    onChange={(value) => handleFilterChange(option.name, value)}
                    onOpenChange={(open) => handleDropdownVisibleChange(open, option)}
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
                    loading={isOptionLoading}
                    options={searchOptionsMap[option.name] || []}
                    allowClear
                    notFoundContent={isOptionLoading ? "Loading..." : "No results found"}
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
                ) : option.customComponent ? (
                  (() => {
                    const CustomComponent = option.customComponent;
                    return (
                      <CustomComponent
                        value={tempValues[option.name] || undefined}
                        onChange={(value) => handleFilterChange(option.name, value ?? "")}
                        placeholder={`Select ${option.label || option.name}...`}
                        allFilters={tempValues}
                      />
                    );
                  })()
                ) : (
                  <Input
                    className="w-full"
                    placeholder={`Enter ${option.label || option.name}...`}
                    value={tempValues[option.name] || ""}
                    onChange={(e) => handleFilterChange(option.name, e.target.value)}
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
