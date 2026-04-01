import React, { useState, useEffect, useMemo, useRef } from "react";
import { Card, Text } from "@tremor/react";

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
}

const ModelFilters: React.FC<ModelFiltersProps> = ({
  modelHubData,
  onFilteredDataChange,
  showFiltersCard = true,
  className = "",
}) => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [selectedProvider, setSelectedProvider] = useState<string>("");
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
    const modes = new Set<string>();
    data.forEach((model) => {
      if (model.mode) modes.add(model.mode);
    });
    return Array.from(modes);
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
        const matchesSearch = model.model_group.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesProvider = selectedProvider === "" || model.providers.includes(selectedProvider);
        const matchesMode = selectedMode === "" || model.mode === selectedMode;

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
    setSelectedProvider("");
    setSelectedMode("");
    setSelectedFeature("");
  };

  // Expose filter values and reset function
  const filterValues = {
    searchTerm,
    selectedProvider,
    selectedMode,
    selectedFeature,
    resetFilters,
  };

  const filtersContent = (
    <div className="flex flex-wrap gap-4 items-center">
      <div>
        <Text className="text-sm font-medium mb-2">Search Models:</Text>
        <input
          type="text"
          placeholder="Search model names..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="border rounded px-3 py-2 w-64 h-10 text-sm"
        />
      </div>
      <div>
        <Text className="text-sm font-medium mb-2">Provider:</Text>
        <select
          value={selectedProvider}
          onChange={(e) => setSelectedProvider(e.target.value)}
          className="border rounded px-3 py-2 text-sm text-gray-600 w-40 h-10"
        >
          <option value="" className="text-sm text-gray-600">
            All Providers
          </option>
          {modelHubData &&
            getUniqueProviders(modelHubData).map((provider) => (
              <option key={provider} value={provider} className="text-sm text-gray-800">
                {provider}
              </option>
            ))}
        </select>
      </div>
      <div>
        <Text className="text-sm font-medium mb-2">Mode:</Text>
        <select
          value={selectedMode}
          onChange={(e) => setSelectedMode(e.target.value)}
          className="border rounded px-3 py-2 text-sm text-gray-600 w-32 h-10"
        >
          <option value="" className="text-sm text-gray-600">
            All Modes
          </option>
          {modelHubData &&
            getUniqueModes(modelHubData).map((mode) => (
              <option key={mode} value={mode} className="text-sm text-gray-800">
                {mode}
              </option>
            ))}
        </select>
      </div>
      <div>
        <Text className="text-sm font-medium mb-2">Features:</Text>
        <select
          value={selectedFeature}
          onChange={(e) => setSelectedFeature(e.target.value)}
          className="border rounded px-3 py-2 text-sm text-gray-600 w-48 h-10"
        >
          <option value="" className="text-sm text-gray-600">
            All Features
          </option>
          {modelHubData &&
            getUniqueFeatures(modelHubData).map((feature) => (
              <option key={feature} value={feature} className="text-sm text-gray-800">
                {feature}
              </option>
            ))}
        </select>
      </div>

      {/* Clear filters button */}
      {(searchTerm || selectedProvider || selectedMode || selectedFeature) && (
        <div className="flex items-end">
          <button
            onClick={resetFilters}
            className="text-blue-600 hover:text-blue-800 text-sm underline h-10 flex items-center"
          >
            Clear Filters
          </button>
        </div>
      )}
    </div>
  );

  if (showFiltersCard) {
    return <Card className={`mb-6 ${className}`}>{filtersContent}</Card>;
  }

  return <div className={className}>{filtersContent}</div>;
};

export default ModelFilters;
