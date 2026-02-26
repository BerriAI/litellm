import React, { useState, useMemo } from "react";
import { Input } from "antd";
import { ArrowLeftOutlined, SearchOutlined } from "@ant-design/icons";
import { ProviderGroup, ModelInfo, formatCost, formatContextWindow, getCapabilities } from "./model_garden_utils";

interface ModelGardenDetailViewProps {
  group: ProviderGroup;
  onBack: () => void;
}

const ModelGardenDetailView: React.FC<ModelGardenDetailViewProps> = ({
  group,
  onBack,
}) => {
  const [activeTab, setActiveTab] = useState("overview");
  const [modelSearch, setModelSearch] = useState("");

  const chatModels = useMemo(() => {
    return group.models
      .filter((m) => m.mode === "chat" || m.mode === "responses")
      .filter((m) => {
        if (!modelSearch) return true;
        return m.key.toLowerCase().includes(modelSearch.toLowerCase());
      });
  }, [group.models, modelSearch]);

  const otherModels = useMemo(() => {
    return group.models
      .filter((m) => m.mode !== "chat" && m.mode !== "responses")
      .filter((m) => {
        if (!modelSearch) return true;
        return m.key.toLowerCase().includes(modelSearch.toLowerCase());
      });
  }, [group.models, modelSearch]);

  const detailRows = [
    { property: "Provider", value: group.displayName },
    { property: "Total Models", value: String(group.modelCount) },
    { property: "Supported Modes", value: group.modes.join(", ") || "-" },
    { property: "Capabilities", value: group.capabilities.join(", ") || "-" },
  ];

  const tabs = [
    { key: "overview", label: "Overview" },
    { key: "models", label: `Models (${group.modelCount})` },
  ];

  return (
    <div style={{ maxWidth: 960, margin: "0 auto" }}>
      <div
        onClick={onBack}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          color: "#5f6368",
          cursor: "pointer",
          fontSize: 14,
          marginBottom: 24,
        }}
      >
        <ArrowLeftOutlined style={{ fontSize: 11 }} />
        <span>Back to Model Garden</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
        {group.logo && (
          <img
            src={group.logo}
            alt=""
            style={{ width: 40, height: 40, borderRadius: 8, objectFit: "contain" }}
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        )}
        <h1 style={{ fontSize: 28, fontWeight: 400, color: "#202124", margin: 0, lineHeight: 1.2 }}>
          {group.displayName}
        </h1>
      </div>

      <p style={{ fontSize: 14, color: "#5f6368", margin: "0 0 20px 0", lineHeight: 1.6 }}>
        {group.modelCount} models available across {group.modes.length} mode{group.modes.length !== 1 ? "s" : ""}: {group.modes.join(", ")}
      </p>

      <div style={{ borderBottom: "1px solid #dadce0", marginBottom: 28 }}>
        <div style={{ display: "flex", gap: 0 }}>
          {tabs.map((tab) => (
            <div
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                padding: "12px 20px",
                fontSize: 14,
                color: activeTab === tab.key ? "#1a73e8" : "#5f6368",
                borderBottom: activeTab === tab.key ? "3px solid #1a73e8" : "3px solid transparent",
                cursor: "pointer",
                fontWeight: activeTab === tab.key ? 500 : 400,
                marginBottom: -1,
              }}
            >
              {tab.label}
            </div>
          ))}
        </div>
      </div>

      {activeTab === "overview" && (
        <div style={{ display: "flex", gap: 64 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 12px 0" }}>Overview</h2>
            <p style={{ fontSize: 14, color: "#3c4043", lineHeight: 1.7, margin: "0 0 32px 0" }}>
              {group.displayName} provides {group.modelCount} models on LiteLLM.
              Available modes include {group.modes.join(", ")}.
            </p>

            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 4px 0" }}>Provider Details</h2>
            <p style={{ fontSize: 13, color: "#5f6368", margin: "0 0 16px 0" }}>Details are as follows</p>

            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #dadce0" }}>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500, width: 200 }}>
                    Property
                  </th>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500 }}>
                    {group.displayName}
                  </th>
                </tr>
              </thead>
              <tbody>
                {detailRows.map((row, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #f1f3f4" }}>
                    <td style={{ padding: "12px 0", color: "#3c4043" }}>{row.property}</td>
                    <td style={{ padding: "12px 0", color: "#202124" }}>{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ width: 240, flexShrink: 0 }}>
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>Provider ID</div>
              <div style={{ fontSize: 13, color: "#202124", wordBreak: "break-all" }}>
                {group.provider}
              </div>
            </div>

            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>Model Count</div>
              <div style={{ fontSize: 13, color: "#202124" }}>{group.modelCount}</div>
            </div>

            {group.capabilities.length > 0 && (
              <div style={{ marginBottom: 28 }}>
                <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 8 }}>Capabilities</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {group.capabilities.map((cap) => (
                    <span
                      key={cap}
                      style={{
                        fontSize: 12,
                        padding: "4px 12px",
                        borderRadius: 16,
                        border: "1px solid #dadce0",
                        color: "#3c4043",
                        backgroundColor: "#fff",
                      }}
                    >
                      {cap}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "models" && (
        <div>
          <div style={{ marginBottom: 16 }}>
            <Input
              placeholder="Search models..."
              prefix={<SearchOutlined style={{ color: "#9ca3af" }} />}
              value={modelSearch}
              onChange={(e) => setModelSearch(e.target.value)}
              style={{ maxWidth: 400, borderRadius: 8 }}
            />
          </div>

          {chatModels.length > 0 && (
            <>
              <h3 style={{ fontSize: 16, fontWeight: 500, color: "#202124", margin: "0 0 12px 0" }}>
                Chat & Response Models ({chatModels.length})
              </h3>
              <ModelTable models={chatModels} />
            </>
          )}

          {otherModels.length > 0 && (
            <>
              <h3 style={{ fontSize: 16, fontWeight: 500, color: "#202124", margin: "24px 0 12px 0" }}>
                Other Models ({otherModels.length})
              </h3>
              <ModelTable models={otherModels} />
            </>
          )}
        </div>
      )}
    </div>
  );
};

const ModelTable: React.FC<{ models: ModelInfo[] }> = ({ models }) => {
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
      <thead>
        <tr style={{ backgroundColor: "#f8f9fa", borderBottom: "1px solid #dadce0" }}>
          <th style={{ textAlign: "left", padding: "10px 12px", color: "#5f6368", fontWeight: 500 }}>Model</th>
          <th style={{ textAlign: "left", padding: "10px 12px", color: "#5f6368", fontWeight: 500 }}>Mode</th>
          <th style={{ textAlign: "right", padding: "10px 12px", color: "#5f6368", fontWeight: 500 }}>Input Cost</th>
          <th style={{ textAlign: "right", padding: "10px 12px", color: "#5f6368", fontWeight: 500 }}>Output Cost</th>
          <th style={{ textAlign: "right", padding: "10px 12px", color: "#5f6368", fontWeight: 500 }}>Context</th>
          <th style={{ textAlign: "left", padding: "10px 12px", color: "#5f6368", fontWeight: 500 }}>Capabilities</th>
        </tr>
      </thead>
      <tbody>
        {models.slice(0, 100).map((model, i) => {
          const caps = getCapabilities(model);
          return (
            <tr key={model.key} style={{ borderBottom: "1px solid #f1f3f4" }}>
              <td style={{ padding: "10px 12px", color: "#202124", fontFamily: "monospace", fontSize: 12 }}>
                {model.key}
              </td>
              <td style={{ padding: "10px 12px", color: "#3c4043" }}>{model.mode}</td>
              <td style={{ padding: "10px 12px", color: "#202124", textAlign: "right" }}>
                {formatCost(model.input_cost_per_token)}
              </td>
              <td style={{ padding: "10px 12px", color: "#202124", textAlign: "right" }}>
                {formatCost(model.output_cost_per_token)}
              </td>
              <td style={{ padding: "10px 12px", color: "#202124", textAlign: "right" }}>
                {formatContextWindow(model.max_input_tokens)}
              </td>
              <td style={{ padding: "10px 12px" }}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                  {caps.slice(0, 3).map((c) => (
                    <span
                      key={c}
                      style={{
                        fontSize: 10,
                        padding: "1px 6px",
                        borderRadius: 8,
                        backgroundColor: "#f0f9ff",
                        color: "#1d4ed8",
                      }}
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
};

export default ModelGardenDetailView;
