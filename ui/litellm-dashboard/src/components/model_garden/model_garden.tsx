import React, { useState, useMemo } from "react";
import { Input, Spin } from "antd";
import { SearchOutlined, ArrowRightOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { ProviderCard, NewModelCard } from "./model_garden_card";
import ModelGardenDetailView from "./model_garden_detail";
import {
  ProviderGroup,
  ModelInfo,
  parseModelCostMap,
  detectNewModels,
  getProviderDisplayName,
  getProviderLogo,
} from "./model_garden_utils";

interface ModelGardenProps {
  modelCostMap: Record<string, any> | null | undefined;
  isLoading: boolean;
}

const ModelGarden: React.FC<ModelGardenProps> = ({ modelCostMap, isLoading }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<ProviderGroup | null>(null);
  const [showAllProviders, setShowAllProviders] = useState(false);
  const [showAllNew, setShowAllNew] = useState(false);

  const CARDS_PER_ROW = 5;
  const VISIBLE_ROWS = 2;

  const providerGroups = useMemo(() => {
    if (!modelCostMap) return [];
    return parseModelCostMap(modelCostMap);
  }, [modelCostMap]);

  const newModels = useMemo(() => {
    if (!modelCostMap) return [];
    return detectNewModels(modelCostMap);
  }, [modelCostMap]);

  const filteredProviders = useMemo(() => {
    if (!searchQuery) return providerGroups;
    const q = searchQuery.toLowerCase();
    return providerGroups.filter(
      (g) =>
        g.displayName.toLowerCase().includes(q) ||
        g.provider.toLowerCase().includes(q) ||
        g.modes.some((m) => m.toLowerCase().includes(q)) ||
        g.capabilities.some((c) => c.toLowerCase().includes(q)) ||
        g.models.some((m) => m.key.toLowerCase().includes(q))
    );
  }, [providerGroups, searchQuery]);

  const filteredNewModels = useMemo(() => {
    if (!searchQuery) return newModels;
    const q = searchQuery.toLowerCase();
    return newModels.filter(
      (m) =>
        m.key.toLowerCase().includes(q) ||
        m.litellm_provider.toLowerCase().includes(q)
    );
  }, [newModels, searchQuery]);

  if (selectedProvider) {
    return (
      <ModelGardenDetailView
        group={selectedProvider}
        onBack={() => setSelectedProvider(null)}
      />
    );
  }

  if (isLoading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
        <p style={{ marginTop: 16, color: "#6b7280" }}>Loading model catalog...</p>
      </div>
    );
  }

  const totalModels = providerGroups.reduce((sum, g) => sum + g.modelCount, 0);

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Input
          size="large"
          placeholder="Search models, providers, or capabilities..."
          prefix={<SearchOutlined style={{ color: "#9ca3af" }} />}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{ borderRadius: 8 }}
        />
      </div>

      {/* What's New Section */}
      {filteredNewModels.length > 0 && (
        <div style={{ marginBottom: 40 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
            <h2 style={{ fontSize: 20, fontWeight: 600, color: "#111827", margin: 0, display: "flex", alignItems: "center", gap: 8 }}>
              <ThunderboltOutlined style={{ color: "#f59e0b" }} />
              {"What's New"}
            </h2>
            {filteredNewModels.length > CARDS_PER_ROW * VISIBLE_ROWS && (
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: 14,
                  color: "#1a73e8",
                  cursor: "pointer",
                }}
                onClick={() => setShowAllNew(!showAllNew)}
              >
                {showAllNew ? (
                  <>Show less</>
                ) : (
                  <>
                    <ArrowRightOutlined style={{ fontSize: 12 }} />
                    {`Show all (${filteredNewModels.length})`}
                  </>
                )}
              </span>
            )}
          </div>
          <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
            Recently added models detected from the model pricing catalog.
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: 16,
            }}
          >
            {(showAllNew
              ? filteredNewModels
              : filteredNewModels.slice(0, CARDS_PER_ROW * VISIBLE_ROWS)
            ).map((model) => (
              <NewModelCard
                key={model.key}
                model={model}
                providerLogo={getProviderLogo(model.litellm_provider)}
                providerName={getProviderDisplayName(model.litellm_provider)}
                onClick={() => {
                  const group = providerGroups.find(
                    (g) => g.provider === model.litellm_provider
                  );
                  if (group) setSelectedProvider(group);
                }}
              />
            ))}
          </div>
        </div>
      )}

      {/* All Providers Section */}
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
          <h2 style={{ fontSize: 20, fontWeight: 600, color: "#111827", margin: 0 }}>
            All Providers
          </h2>
          {filteredProviders.length > CARDS_PER_ROW * VISIBLE_ROWS && (
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                fontSize: 14,
                color: "#1a73e8",
                cursor: "pointer",
              }}
              onClick={() => setShowAllProviders(!showAllProviders)}
            >
              {showAllProviders ? (
                <>Show less</>
              ) : (
                <>
                  <ArrowRightOutlined style={{ fontSize: 12 }} />
                  {`Show all (${filteredProviders.length})`}
                </>
              )}
            </span>
          )}
        </div>
        <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
          {totalModels.toLocaleString()} models from {providerGroups.length} providers.
          All data sourced from the LiteLLM model pricing catalog.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: 16,
          }}
        >
          {(showAllProviders
            ? filteredProviders
            : filteredProviders.slice(0, CARDS_PER_ROW * VISIBLE_ROWS)
          ).map((group) => (
            <ProviderCard
              key={group.provider}
              group={group}
              onClick={() => setSelectedProvider(group)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelGarden;
