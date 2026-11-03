import React, { useState, useEffect, useMemo, useRef } from "react";
import { Search } from "lucide-react";
import ModelHubProviders from "@/components/AIHub/ModelHubProviders";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { displayMode } from "@/lib/decisionModels";
import { matchesSearchTerm } from "@/utils/searchUtils";

interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  mode?: string;
  tpm?: number;
  rpm?: number;
  supports_parallel_function_calling: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supported_openai_params?: string[];
  is_public_model_group: boolean;
  [key: string]: any;
}

interface ModelFiltersProps {
  modelHubData: ModelGroupInfo[];
  onFilteredDataChange: (filteredData: ModelGroupInfo[]) => void;
  showFiltersCard?: boolean;
  className?: string;
  extraModes?: readonly string[];
  onModeChange?: (mode: string) => void;
  selectedProvider?: string;
  onProviderChange?: (provider: string) => void;
  showProviderFilter?: boolean;
  variant?: "dashboard" | "chat";
}

interface ChatFilterSelectProps {
  label: string;
  emptyLabel: string;
  options: string[];
  value: string;
  onValueChange: (value: string) => void;
}

function ChatFilterSelect({ label, emptyLabel, options, value, onValueChange }: ChatFilterSelectProps) {
  const items = [{ value: "", label: emptyLabel }, ...options.map((option) => ({ value: option, label: option }))];
  return (
    <Select value={value} items={items} onValueChange={(next) => onValueChange(next ?? "")}>
      <SelectTrigger aria-label={label} className="chat-model-filter-select">
        <SelectValue />
      </SelectTrigger>
      <SelectContent className="chat-theme">
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

const ModelFilters: React.FC<ModelFiltersProps> = ({
  modelHubData,
  onFilteredDataChange,
  showFiltersCard = true,
  className = "",
  extraModes = [],
  onModeChange,
  selectedProvider: controlledProvider,
  onProviderChange,
  showProviderFilter = true,
  variant = "dashboard",
}) => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [internalProvider, setInternalProvider] = useState<string>("");
  const selectedProvider = controlledProvider ?? internalProvider;
  const [selectedMode, setSelectedMode] = useState<string>("");
  const [selectedFeature, setSelectedFeature] = useState<string>("");
  const previousFilteredDataRef = useRef<ModelGroupInfo[]>([]);

  // Helper functions to get unique values
  const getUniqueProviders = (data: ModelGroupInfo[]) => {
    const providers = new Set<string>();
    data.forEach((model) => {
      model.providers.forEach((provider) => providers.add(provider));
    });
    return Array.from(providers);
  };

  const getUniqueModes = (data: ModelGroupInfo[]) => {
    const modes = new Set<string>(extraModes);
    data.forEach((model) => {
      const mode = displayMode(model.mode);
      if (mode) modes.add(mode);
    });
    return Array.from(modes);
  };

  const changeMode = (mode: string) => {
    setSelectedMode(mode);
    onModeChange?.(mode);
  };

  const changeProvider = (provider: string) => {
    if (controlledProvider === undefined) setInternalProvider(provider);
    onProviderChange?.(provider);
  };

  const getUniqueFeatures = (data: ModelGroupInfo[]) => {
    const features = new Set<string>();
    data.forEach((model) => {
      Object.entries(model)
        .filter(([key, value]) => key.startsWith("supports_") && value === true)
        .forEach(([key]) => {
          const featureName = key
            .replace(/^supports_/, "")
            .split("_")
            .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
            .join(" ");
          features.add(featureName);
        });
    });
    return Array.from(features).sort();
  };

  // Memoized filtered data
  const filteredData = useMemo(() => {
    return (
      modelHubData?.filter((model) => {
        const matchesSearch = matchesSearchTerm(searchTerm, [model.model_group]);
        const matchesProvider = selectedProvider === "" || model.providers.includes(selectedProvider);
        const matchesMode = selectedMode === "" || displayMode(model.mode) === selectedMode;

        // Check if model has the selected feature
        const matchesFeature =
          selectedFeature === "" ||
          Object.entries(model)
            .filter(([key, value]) => key.startsWith("supports_") && value === true)
            .some(([key]) => {
              const featureName = key
                .replace(/^supports_/, "")
                .split("_")
                .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                .join(" ");
              return featureName === selectedFeature;
            });

        return matchesSearch && matchesProvider && matchesMode && matchesFeature;
      }) || []
    );
  }, [modelHubData, searchTerm, selectedProvider, selectedMode, selectedFeature]);

  // Update parent component when filtered data changes
  useEffect(() => {
    // Only call the callback if the filtered data actually changed
    const hasChanged =
      filteredData.length !== previousFilteredDataRef.current.length ||
      filteredData.some((model, index) => model.model_group !== previousFilteredDataRef.current[index]?.model_group);

    if (hasChanged) {
      previousFilteredDataRef.current = filteredData;
      onFilteredDataChange(filteredData);
    }
  }, [filteredData, onFilteredDataChange]);

  // Reset filters function
  const resetFilters = () => {
    setSearchTerm("");
    changeProvider("");
    changeMode("");
    setSelectedFeature("");
  };

  const hasActiveFilters = Boolean(searchTerm || selectedProvider || selectedMode || selectedFeature);

  if (variant === "chat") {
    return (
      <div className={`${className} chat-model-filter-bar`}>
        <InputGroup className="chat-model-filter-search">
          <InputGroupAddon aria-hidden="true">
            <Search className="size-4" />
          </InputGroupAddon>
          <InputGroupInput
            aria-label="Search models"
            placeholder="Search model names..."
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
          />
        </InputGroup>
        {showProviderFilter && (
          <ModelHubProviders
            providers={getUniqueProviders(modelHubData)}
            selectedProviders={selectedProvider ? [selectedProvider] : []}
            onSelect={(provider) => changeProvider(provider ?? "")}
          />
        )}
        <ChatFilterSelect
          label="Model type"
          emptyLabel="All types"
          options={getUniqueModes(modelHubData)}
          value={selectedMode}
          onValueChange={changeMode}
        />
        <ChatFilterSelect
          label="Model capability"
          emptyLabel="All capabilities"
          options={getUniqueFeatures(modelHubData)}
          value={selectedFeature}
          onValueChange={setSelectedFeature}
        />
        {hasActiveFilters && (
          <Button variant="ghost" onClick={resetFilters} className="chat-model-filter-clear">
            Clear Filters
          </Button>
        )}
      </div>
    );
  }

  const filtersContent = (
    <div className="flex flex-wrap gap-4 items-center">
      <div>
        <p className="text-sm font-medium mb-2">Search Models:</p>
        <input
          type="text"
          placeholder="Search model names..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="border rounded-sm px-3 py-2 w-64 h-10 text-sm"
        />
      </div>
      {showProviderFilter && (
        <div>
          <p className="text-sm font-medium mb-2">Provider:</p>
          <select
            value={selectedProvider}
            onChange={(e) => changeProvider(e.target.value)}
            className="border rounded-sm px-3 py-2 text-sm text-muted-foreground w-40 h-10"
          >
            <option value="" className="text-sm text-muted-foreground">
              All Providers
            </option>
            {modelHubData &&
              getUniqueProviders(modelHubData).map((provider) => (
                <option key={provider} value={provider} className="text-sm text-foreground">
                  {provider}
                </option>
              ))}
          </select>
        </div>
      )}
      <div>
        <p className="text-sm font-medium mb-2">Mode:</p>
        <select
          value={selectedMode}
          onChange={(e) => changeMode(e.target.value)}
          className="border rounded-sm px-3 py-2 text-sm text-muted-foreground w-32 h-10"
        >
          <option value="" className="text-sm text-muted-foreground">
            All Modes
          </option>
          {modelHubData &&
            getUniqueModes(modelHubData).map((mode) => (
              <option key={mode} value={mode} className="text-sm text-foreground">
                {mode}
              </option>
            ))}
        </select>
      </div>
      <div>
        <p className="text-sm font-medium mb-2">Features:</p>
        <select
          value={selectedFeature}
          onChange={(e) => setSelectedFeature(e.target.value)}
          className="border rounded-sm px-3 py-2 text-sm text-muted-foreground w-48 h-10"
        >
          <option value="" className="text-sm text-muted-foreground">
            All Features
          </option>
          {modelHubData &&
            getUniqueFeatures(modelHubData).map((feature) => (
              <option key={feature} value={feature} className="text-sm text-foreground">
                {feature}
              </option>
            ))}
        </select>
      </div>

      {/* Clear filters button */}
      {hasActiveFilters && (
        <div className="flex items-end">
          <button
            onClick={resetFilters}
            className="text-info hover:text-info/80 text-sm underline h-10 flex items-center"
          >
            Clear Filters
          </button>
        </div>
      )}
    </div>
  );

  if (showFiltersCard) {
    return <Card className={`mb-6 px-6 ${className}`}>{filtersContent}</Card>;
  }

  return <div className={className}>{filtersContent}</div>;
};

export default ModelFilters;
