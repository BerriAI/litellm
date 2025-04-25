import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button, Input, Dropdown, MenuProps, Select, Spin, Space } from 'antd';
import { Card, Button as TremorButton } from '@tremor/react';
import {
  FilterIcon,
  XIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  SearchIcon
} from '@heroicons/react/outline';
import debounce from 'lodash/debounce';

export interface FilterOption {
  name: string;
  label?: string;
  isSearchable?: boolean;
  searchFn?: (searchText: string) => Promise<Array<{ label: string; value: string }>>;
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
  buttonLabel = "Filter",
}) => {
  const [showFilters, setShowFilters] = useState<boolean>(false);
  const [filterValues, setFilterValues] = useState<FilterValues>(initialValues);
  const [tempValues, setTempValues] = useState<FilterValues>(initialValues);
  const [searchOptionsMap, setSearchOptionsMap] = useState<{ [key: string]: Array<{ label: string; value: string }> }>({});
  const [searchLoadingMap, setSearchLoadingMap] = useState<{ [key: string]: boolean }>({});
  const [searchInputValueMap, setSearchInputValueMap] = useState<{ [key: string]: string }>({});
  
  const filtersRef = useRef<HTMLDivElement>(null);

  const debouncedSearch = useCallback(
    debounce(async (value: string, option: FilterOption) => {
      if (!option.isSearchable || !option.searchFn) return;
      
      setSearchLoadingMap(prev => ({ ...prev, [option.name]: true }));
      try {
        const results = await option.searchFn(value);
        setSearchOptionsMap(prev => ({ ...prev, [option.name]: results }));
      } catch (error) {
        console.error('Error searching:', error);
        setSearchOptionsMap(prev => ({ ...prev, [option.name]: [] }));
      } finally {
        setSearchLoadingMap(prev => ({ ...prev, [option.name]: false }));
      }
    }, 300),
    []
  );

  const handleFilterChange = (name: string, value: string) => {
    setTempValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleApplyFilters = () => {
    setFilterValues(tempValues);
    onApplyFilters(tempValues);
    // setShowFilters(false);
  };

  const resetFilters = () => {
    const emptyValues: FilterValues = {};
    options.forEach(option => {
      emptyValues[option.name] = '';
    });
    setTempValues(emptyValues);
    setFilterValues(emptyValues);
    setSearchInputValueMap({});
    setSearchOptionsMap({});
    onResetFilters();
  };

  return (
    <div className="relative" ref={filtersRef}>
      <TremorButton
        icon={FilterIcon}
        onClick={() => setShowFilters(!showFilters)}
        variant="secondary"
        size='xs'
        className="flex items-center pr-2"
      >
        {buttonLabel}
      </TremorButton>
      {showFilters && (
        <div className="mt-2">
          <Space size="middle" align="center">
            {options.map((option) => (
              <div key={option.name} className="flex items-center gap-2">
                {option.isSearchable ? (
                  <Select
                    showSearch
                    style={{ width: 200 }}
                    placeholder={`Search ${option.label || option.name}...`}
                    value={tempValues[option.name] || undefined}
                    onChange={(value) => handleFilterChange(option.name, value)}
                    onSearch={(value) => {
                      setSearchInputValueMap(prev => ({ ...prev, [option.name]: value }));
                      if (option.searchFn) {
                        debouncedSearch(value, option);
                      }
                    }}
                    filterOption={false}
                    loading={searchLoadingMap[option.name]}
                    options={searchOptionsMap[option.name] || []}
                    allowClear
                  />
                ) : (
                  <Input
                    style={{ width: 200 }}
                    placeholder={`Enter ${option.label || option.name}...`}
                    value={tempValues[option.name] || ''}
                    onChange={(e) => handleFilterChange(option.name, e.target.value)}
                    allowClear
                  />
                )}
              </div>
            ))}
            <Space>
              <Button onClick={resetFilters}>Reset</Button>
              <Button onClick={handleApplyFilters}>Apply</Button>
            </Space>
          </Space>
        </div>
      )}
    </div>
  );
};

export default FilterComponent;