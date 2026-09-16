import React, { useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import {
  filterModelHubData,
  modelFeatureNames,
  type ModelFiltersState,
  type ModelFilterValues,
  useLocalModelFiltersState,
} from "./useModelFiltersState";

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
  onFilteredDataChange?: (filteredData: ModelGroupInfo[]) => void;
  filtersState?: ModelFiltersState;
  showFiltersCard?: boolean;
  className?: string;
}

const getUniqueProviders = (data: ModelGroupInfo[]) => [...new Set(data.flatMap((model) => model.providers))];

const getUniqueModes = (data: ModelGroupInfo[]) =>
  [...new Set(data.map((model) => model.mode))].filter((mode): mode is string => Boolean(mode));

const getUniqueFeatures = (data: ModelGroupInfo[]) => [...new Set(data.flatMap(modelFeatureNames))].sort();

const sameModelGroups = (left: ModelGroupInfo[], right: ModelGroupInfo[]) =>
  left.length === right.length && left.every((model, index) => model.model_group === right[index]?.model_group);

const useFilteredDataNotifier = (
  modelHubData: ModelGroupInfo[],
  values: ModelFilterValues,
  onFilteredDataChange: ModelFiltersProps["onFilteredDataChange"],
) => {
  const previousFilteredDataRef = useRef<ModelGroupInfo[]>([]);

  useEffect(() => {
    if (onFilteredDataChange === undefined) return;
    const filteredData = filterModelHubData(modelHubData ?? [], values);
    if (sameModelGroups(filteredData, previousFilteredDataRef.current)) return;
    previousFilteredDataRef.current = filteredData;
    onFilteredDataChange(filteredData);
  }, [modelHubData, values, onFilteredDataChange]);
};

const ModelFilters: React.FC<ModelFiltersProps> = ({
  modelHubData,
  onFilteredDataChange,
  filtersState,
  showFiltersCard = true,
  className = "",
}) => {
  const localFiltersState = useLocalModelFiltersState();
  const { values, update, reset: resetFilters } = filtersState ?? localFiltersState;
  const { search: searchTerm, provider: selectedProvider, mode: selectedMode, feature: selectedFeature } = values;
  const hasActiveFilters = Object.values(values).some(Boolean);
  useFilteredDataNotifier(modelHubData, values, onFilteredDataChange);

  const filtersContent = (
    <div className="flex flex-wrap gap-4 items-center">
      <div>
        <p className="text-sm font-medium mb-2">Search Models:</p>
        <input
          type="text"
          placeholder="Search model names..."
          value={searchTerm}
          onChange={(e) => update({ search: e.target.value })}
          className="border rounded-sm px-3 py-2 w-64 h-10 text-sm"
        />
      </div>
      <div>
        <p className="text-sm font-medium mb-2">Provider:</p>
        <select
          value={selectedProvider}
          onChange={(e) => update({ provider: e.target.value })}
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
      <div>
        <p className="text-sm font-medium mb-2">Mode:</p>
        <select
          value={selectedMode}
          onChange={(e) => update({ mode: e.target.value })}
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
          onChange={(e) => update({ feature: e.target.value })}
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
