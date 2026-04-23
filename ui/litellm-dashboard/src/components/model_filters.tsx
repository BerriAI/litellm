import React, { useState, useEffect, useMemo, useRef } from "react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

  const getUniqueProviders = (data: ModelGroupInfo[]) => {
    const providers = new Set<string>();
    data.forEach((m) => m.providers.forEach((p) => providers.add(p)));
    return Array.from(providers);
  };

  const getUniqueModes = (data: ModelGroupInfo[]) => {
    const modes = new Set<string>();
    data.forEach((m) => {
      if (m.mode) modes.add(m.mode);
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
            .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");
          features.add(featureName);
        });
    });
    return Array.from(features).sort();
  };

  const filteredData = useMemo(() => {
    return (
      modelHubData?.filter((model) => {
        const matchesSearch = model.model_group
          .toLowerCase()
          .includes(searchTerm.toLowerCase());
        const matchesProvider =
          selectedProvider === "" ||
          model.providers.includes(selectedProvider);
        const matchesMode = selectedMode === "" || model.mode === selectedMode;
        const matchesFeature =
          selectedFeature === "" ||
          Object.entries(model)
            .filter(
              ([key, value]) => key.startsWith("supports_") && value === true,
            )
            .some(([key]) => {
              const featureName = key
                .replace(/^supports_/, "")
                .split("_")
                .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
                .join(" ");
              return featureName === selectedFeature;
            });
        return (
          matchesSearch && matchesProvider && matchesMode && matchesFeature
        );
      }) || []
    );
  }, [modelHubData, searchTerm, selectedProvider, selectedMode, selectedFeature]);

  useEffect(() => {
    const hasChanged =
      filteredData.length !== previousFilteredDataRef.current.length ||
      filteredData.some(
        (model, index) =>
          model.model_group !==
          previousFilteredDataRef.current[index]?.model_group,
      );
    if (hasChanged) {
      previousFilteredDataRef.current = filteredData;
      onFilteredDataChange(filteredData);
    }
  }, [filteredData, onFilteredDataChange]);

  const resetFilters = () => {
    setSearchTerm("");
    setSelectedProvider("");
    setSelectedMode("");
    setSelectedFeature("");
  };

  const labelClass = "text-sm font-medium mb-2 block";
  const inputClass =
    "border border-input bg-background rounded px-3 py-2 text-sm h-10";

  const filtersContent = (
    <div className="flex flex-wrap gap-4 items-center">
      <div>
        <span className={labelClass}>Search Models:</span>
        <input
          type="text"
          placeholder="Search model names..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className={cn(inputClass, "w-64")}
        />
      </div>
      <div>
        <span className={labelClass}>Provider:</span>
        <select
          value={selectedProvider}
          onChange={(e) => setSelectedProvider(e.target.value)}
          className={cn(inputClass, "text-muted-foreground w-40")}
        >
          <option value="">All Providers</option>
          {modelHubData &&
            getUniqueProviders(modelHubData).map((provider) => (
              <option key={provider} value={provider}>
                {provider}
              </option>
            ))}
        </select>
      </div>
      <div>
        <span className={labelClass}>Mode:</span>
        <select
          value={selectedMode}
          onChange={(e) => setSelectedMode(e.target.value)}
          className={cn(inputClass, "text-muted-foreground w-32")}
        >
          <option value="">All Modes</option>
          {modelHubData &&
            getUniqueModes(modelHubData).map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
        </select>
      </div>
      <div>
        <span className={labelClass}>Features:</span>
        <select
          value={selectedFeature}
          onChange={(e) => setSelectedFeature(e.target.value)}
          className={cn(inputClass, "text-muted-foreground w-48")}
        >
          <option value="">All Features</option>
          {modelHubData &&
            getUniqueFeatures(modelHubData).map((feature) => (
              <option key={feature} value={feature}>
                {feature}
              </option>
            ))}
        </select>
      </div>

      {(searchTerm || selectedProvider || selectedMode || selectedFeature) && (
        <div className="flex items-end">
          <button
            onClick={resetFilters}
            className="text-primary hover:underline text-sm h-10 flex items-center"
          >
            Clear Filters
          </button>
        </div>
      )}
    </div>
  );

  if (showFiltersCard) {
    return <Card className={cn("mb-6 p-4", className)}>{filtersContent}</Card>;
  }

  return <div className={className}>{filtersContent}</div>;
};

export default ModelFilters;
