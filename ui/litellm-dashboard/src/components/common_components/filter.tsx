import React, { useState, useRef, useEffect } from 'react';
import { Button, Input, Dropdown, MenuProps } from 'antd';
import { Card, Button as TremorButton } from '@tremor/react';
import {
  FilterIcon,
  XIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon
} from '@heroicons/react/outline';

interface FilterOption {
  name: string;
  label?: string;
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
  
  const filtersRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (filtersRef.current && 
          !filtersRef.current.contains(target) && 
          !target.closest('.ant-dropdown')) {
        setShowFilters(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
        <Card className="absolute left-0 mt-2 w-96 z-50 border border-gray-200 shadow-lg">
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Where</span>
              
              <Dropdown
                menu={{
                  items: dropdownItems,
                  onClick: ({ key }) => {
                    setSelectedFilter(key);
                    setDropdownOpen(false);
                  }
                }}
                onOpenChange={setDropdownOpen}
                open={dropdownOpen}
                trigger={['click']}
              >
                <Button className="min-w-32 text-left flex justify-between items-center">
                  {selectedFilter}
                  {dropdownOpen ? (
                    <ChevronUpIcon className="h-4 w-4" />
                  ) : (
                    <ChevronDownIcon className="h-4 w-4" />
                  )}
                </Button>
              </Dropdown>
              
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
            </div>
            
            <div className="flex justify-end gap-2">
              <Button
                onClick={() => {
                  clearFilters();
                  const emptyValues: FilterValues = {};
                  options.forEach(option => {
                    emptyValues[option.name] = '';
                  });
                  onResetFilters();
                  setShowFilters(false);
                }}
              >
                Cancel
              </Button>
              <Button type="primary" onClick={handleApplyFilters}>
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