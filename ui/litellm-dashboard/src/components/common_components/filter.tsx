import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button, Input, Dropdown, MenuProps, Select, Spin } from 'antd';
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
  const [selectedFilter, setSelectedFilter] = useState<string>(options[0]?.name || '');
  const [filterValues, setFilterValues] = useState<FilterValues>(initialValues);
  const [tempValues, setTempValues] = useState<FilterValues>(initialValues);
  const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
  const [searchOptions, setSearchOptions] = useState<Array<{ label: string; value: string }>>([]);
  const [searchLoading, setSearchLoading] = useState<boolean>(false);
  const [searchInputValue, setSearchInputValue] = useState<string>('');
  
  const filtersRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (filtersRef.current && 
          !filtersRef.current.contains(target) && 
          !target.closest('.ant-dropdown') &&
          !target.closest('.ant-select-dropdown')) {
        setShowFilters(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (options.length > 0 && options[0].isSearchable && options[0].searchFn) {
      loadInitialOptions(options[0]);
    }
  }, []);

  const loadInitialOptions = async (option: FilterOption) => {
    if (!option.isSearchable || !option.searchFn) return;
    
    setSearchLoading(true);
    try {
      const results = await option.searchFn('');
      setSearchOptions(results);
    } catch (error) {
      console.error('Error loading initial options:', error);
      setSearchOptions([]);
    } finally {
      setSearchLoading(false);
    }
  };
  
  useEffect(() => {
    if (showFilters && currentOption?.isSearchable && currentOption?.searchFn) {
      loadInitialOptions(currentOption);
    }
  }, [showFilters, selectedFilter]);

  const handleFilterSelect = (key: string) => {
    setSelectedFilter(key);
    setDropdownOpen(false);
    
    const newOption = options.find(opt => opt.name === key);
    if (newOption?.isSearchable && newOption?.searchFn) {
      loadInitialOptions(newOption);
    } else {
      setSearchOptions([]);
    }
  };

  const debouncedSearch = useCallback(
    debounce(async (value: string, option: FilterOption) => {
      if (!option.isSearchable || !option.searchFn) return;
      
      setSearchLoading(true);
      try {
        const results = await option.searchFn(value);
        setSearchOptions(results);
      } catch (error) {
        console.error('Error searching:', error);
        setSearchOptions([]);
      } finally {
        setSearchLoading(false);
      }
    }, 300),
    []
  );
  
  const handleFilterChange = (value: string) => {
    setTempValues(prev => ({
      ...prev,
      [selectedFilter]: value
    }));
  };
  
  const clearFilters = () => {
    const emptyValues: FilterValues = {};
    options.forEach(option => {
      emptyValues[option.name] = '';
    });
    setTempValues(emptyValues);
  };
  
  const handleApplyFilters = () => {
    setFilterValues(tempValues);
    onApplyFilters(tempValues);
    setShowFilters(false);
  };
  
  const dropdownItems: MenuProps['items'] = options.map(option => ({
    key: option.name,
    label: (
      <div className="flex items-center gap-2">
        {selectedFilter === option.name && (
          <CheckIcon className="h-4 w-4 text-blue-600" />
        )}
        {option.label || option.name}
      </div>
    ),
  }));

  const currentOption = options.find(option => option.name === selectedFilter);

  const resetFilters = () => {
    const emptyValues: FilterValues = {};
    options.forEach(option => {
      emptyValues[option.name] = '';
    });
    setTempValues(emptyValues);
    setFilterValues(emptyValues);
    setSearchInputValue('');
    setSearchOptions([]);
    onResetFilters(); // Call the parent's reset function
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
        <Card className="absolute left-0 mt-2 w-[500px] z-50 border border-gray-200 shadow-lg">
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Where</span>
              
              <Dropdown
                menu={{
                  items: dropdownItems,
                  onClick: ({ key }) => handleFilterSelect(key),
                  style: { minWidth: '200px' }
                }}
                onOpenChange={setDropdownOpen}
                open={dropdownOpen}
                trigger={['click']}
              >
                <Button className="min-w-40 text-left flex justify-between items-center">
                  {currentOption?.label || selectedFilter}
                  {dropdownOpen ? (
                    <ChevronUpIcon className="h-4 w-4" />
                  ) : (
                    <ChevronDownIcon className="h-4 w-4" />
                  )}
                </Button>
              </Dropdown>
              
              {currentOption?.isSearchable ? (
              <Select
                showSearch
                placeholder={`Search ${currentOption.label || selectedFilter}...`}
                value={tempValues[selectedFilter] || undefined}
                onChange={(value) => handleFilterChange(value)}
                onSearch={(value) => {
                  setSearchInputValue(value);
                  debouncedSearch(value, currentOption);
                }}
                onInputKeyDown={(e) => {
                  if (e.key === 'Enter' && searchInputValue) {
                    // Allow manual entry of the value on Enter
                    handleFilterChange(searchInputValue);
                    e.preventDefault();
                  }
                }}
                filterOption={false}
                className="flex-1 w-full max-w-full truncate min-w-100"
                loading={searchLoading}
                options={searchOptions}
                allowClear
                notFoundContent={
                  searchLoading ? <Spin size="small" /> : (
                    <div className="p-2">
                      {searchInputValue && (
                        <Button 
                          type="link" 
                          className="p-0 mt-1"
                          onClick={() => {
                            handleFilterChange(searchInputValue);
                            // Close the dropdown/select after selecting the value
                            const selectElement = document.activeElement as HTMLElement;
                            if (selectElement) {
                              selectElement.blur();
                            }
                          }}
                        >
                          Use &ldquo;{searchInputValue}&rdquo; as filter value
                        </Button>
                        )}
                    </div>
                  )
                }
              />
            ) : (
                <Input
                  placeholder="Enter value..."
                  value={tempValues[selectedFilter] || ''}
                  onChange={(e) => handleFilterChange(e.target.value)}
                  className="px-3 py-1.5 border rounded-md text-sm flex-1 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  suffix={
                    tempValues[selectedFilter] ? (
                      <XIcon
                        className="h-4 w-4 cursor-pointer text-gray-400 hover:text-gray-500"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleFilterChange('');
                        }}
                      />
                    ) : null
                  }
                />
              )}
            </div>


            <div className="flex gap-2 justify-end">
              <Button
                onClick={() => {
                  clearFilters();
                  onResetFilters();
                  setShowFilters(false);
                }}
              >
                Reset
              </Button>
              <Button onClick={handleApplyFilters}>
                Apply Filters
              </Button>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default FilterComponent;