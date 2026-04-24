import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { Filter, Loader2, X } from "lucide-react";
import debounce from "lodash/debounce";
import React, { useCallback, useEffect, useRef, useState } from "react";

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

// Searchable combobox — shadcn Popover + async results list. Mirrors antd's
// Select in showSearch mode closely enough to preserve UX:
//   - typing triggers debounced search via option.searchFn
//   - selecting a result closes the popover and sets the value
//   - clear button (x) on the trigger resets the value
interface SearchableSelectProps {
  option: FilterOption;
  value: string;
  loading: boolean;
  results: Array<{ label: string; value: string }>;
  onOpenChange: (open: boolean) => void;
  onSearch: (text: string) => void;
  onSelect: (value: string) => void;
}

const SearchableSelect: React.FC<SearchableSelectProps> = ({
  option,
  value,
  loading,
  results,
  onOpenChange,
  onSearch,
  onSelect,
}) => {
  const [open, setOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const selectedOption = results.find((r) => r.value === value);
  const displayLabel = selectedOption?.label ?? value;

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    onOpenChange(next);
    if (!next) setSearchText("");
  };

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          className={cn(
            "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          )}
          aria-label={`Search ${option.label || option.name}...`}
        >
          <span className={cn("truncate", !value && "text-muted-foreground")}>
            {value
              ? displayLabel
              : `Search ${option.label || option.name}...`}
          </span>
          <span className="flex items-center gap-1 text-muted-foreground">
            {value && (
              <span
                role="button"
                tabIndex={0}
                aria-label="Clear"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect("");
                }}
                className="inline-flex items-center"
              >
                <X className="h-3 w-3" />
              </span>
            )}
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[--radix-popover-trigger-width] max-w-none">
        <div className="flex items-center border-b border-border p-2">
          <Input
            value={searchText}
            onChange={(e) => {
              setSearchText(e.target.value);
              onSearch(e.target.value);
            }}
            placeholder={`Search ${option.label || option.name}...`}
            className="h-8 border-0 focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none"
          />
        </div>
        <div className="max-h-60 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center p-3 text-sm text-muted-foreground gap-2">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading...
            </div>
          ) : results.length === 0 ? (
            <div className="p-3 text-sm text-muted-foreground text-center">
              No results found
            </div>
          ) : (
            results.map((r) => (
              <button
                key={r.value}
                type="button"
                onClick={() => {
                  onSelect(r.value);
                  handleOpenChange(false);
                }}
                className={cn(
                  "w-full text-left text-sm px-3 py-2 hover:bg-muted",
                  r.value === value && "bg-muted font-medium",
                )}
              >
                {r.label}
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};

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
  const [initialOptionsLoaded, setInitialOptionsLoaded] = useState<{
    [key: string]: boolean;
  }>({});

  // Keep internal state in sync with externally provided initialValues
  const initialValuesRef = useRef(initialValues);
  useEffect(() => {
    initialValuesRef.current = initialValues;
  }, [initialValues]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
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
    [],
  );

  // Load initial options for searchable filters
  const loadInitialOptions = useCallback(
    async (option: FilterOption) => {
      if (!option.isSearchable || !option.searchFn || initialOptionsLoaded[option.name]) return;

      setSearchLoadingMap((prev) => ({ ...prev, [option.name]: true }));
      setInitialOptionsLoaded((prev) => ({ ...prev, [option.name]: true }));

      try {
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

  const handleDropdownOpenChange = (open: boolean, option: FilterOption) => {
    if (open && option.isSearchable && !initialOptionsLoaded[option.name]) {
      loadInitialOptions(option);
    }
  };

  const orderedFilters = [
    "Team ID",
    "Status",
    "Organization ID",
    "Key Alias",
    "User ID",
    "End User",
    "Error Code",
    "Error Message",
    "Key Hash",
    "Model",
  ];

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-6">
        <Button variant="outline" onClick={() => setShowFilters(!showFilters)}>
          <Filter className="h-4 w-4" />
          {buttonLabel}
        </Button>
        <Button variant="outline" onClick={resetFilters}>
          Reset Filters
        </Button>
      </div>

      {showFilters && (
        <div className="grid grid-cols-3 gap-x-6 gap-y-4 mb-6">
          {orderedFilters.map((filterName) => {
            const option = options.find(
              (opt) => opt.label === filterName || opt.name === filterName,
            );
            if (!option) return null;

            return (
              <div key={option.name} className="flex flex-col gap-2">
                <label className="text-sm text-muted-foreground">
                  {option.label || option.name}
                </label>
                {option.isSearchable ? (
                  <SearchableSelect
                    option={option}
                    value={tempValues[option.name] || ""}
                    loading={!!searchLoadingMap[option.name]}
                    results={searchOptionsMap[option.name] || []}
                    onOpenChange={(open) =>
                      handleDropdownOpenChange(open, option)
                    }
                    onSearch={(text) => {
                      if (option.searchFn) debouncedSearch(text, option);
                    }}
                    onSelect={(value) => handleFilterChange(option.name, value)}
                  />
                ) : option.options ? (
                  <Select
                    value={tempValues[option.name] || ""}
                    onValueChange={(value) =>
                      handleFilterChange(option.name, value === "__clear__" ? "" : value)
                    }
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue
                        placeholder={`Select ${option.label || option.name}...`}
                      />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__clear__">
                        <span className="text-muted-foreground">
                          (Any)
                        </span>
                      </SelectItem>
                      {option.options.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value}>
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
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
                    onChange={(e) =>
                      handleFilterChange(option.name, e.target.value)
                    }
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
