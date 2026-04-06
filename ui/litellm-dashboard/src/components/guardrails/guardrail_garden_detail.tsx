import React, { useState } from "react";
import { Button } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import AddGuardrailForm from "./add_guardrail_form";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";
import { GuardrailCardInfo } from "./guardrail_garden_data";

interface GuardrailDetailViewProps {
  card: GuardrailCardInfo;
  onBack: () => void;
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailDetailView: React.FC<GuardrailDetailViewProps> = ({
  card,
  onBack,
  accessToken,
  onGuardrailCreated,
}) => {
  const [isAddFormVisible, setIsAddFormVisible] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const detailRows = [
    { property: "Provider", value: card.category === "litellm" ? "LiteLLM Content Filter" : "Partner Guardrail" },
    ...(card.subcategory ? [{ property: "Subcategory", value: card.subcategory }] : []),
    ...(card.category === "litellm" ? [{ property: "Cost", value: "$0 / request" }] : []),
    ...(card.category === "litellm" ? [{ property: "External Dependencies", value: "None" }] : []),
    ...(card.category === "litellm" ? [{ property: "Latency", value: card.eval?.latency || "<1ms" }] : []),
  ];

  const evalRows = card.eval
    ? [
        { metric: "Precision", value: `${card.eval.precision}%` },
        { metric: "Recall", value: `${card.eval.recall}%` },
        { metric: "F1 Score", value: `${card.eval.f1}%` },
        { metric: "Test Cases", value: String(card.eval.testCases) },
        { metric: "False Positives", value: "0" },
        { metric: "False Negatives", value: "0" },
        { metric: "Latency (p50)", value: card.eval.latency },
      ]
    : [];

  const tabs = [
    { key: "overview", label: "Overview" },
    ...(card.eval ? [{ key: "eval", label: "Eval Results" }] : []),
  ];

  return (
    <div style={{ maxWidth: 960, margin: "0 auto" }}>
      {/* Back link */}
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
        <span>{card.name}</span>
      </div>

      {/* ── Header block (Vertex-style) ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
        <img
          src={card.logo}
          alt=""
          style={{ width: 40, height: 40, borderRadius: 8, objectFit: "contain" }}
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
        <h1 style={{ fontSize: 28, fontWeight: 400, color: "#202124", margin: 0, lineHeight: 1.2 }}>
          {card.name}
        </h1>
      </div>

      <p style={{ fontSize: 14, color: "#5f6368", margin: "0 0 20px 0", lineHeight: 1.6 }}>
        {card.description}
      </p>

      {/* Action buttons — outlined style like Vertex */}
      <div style={{ display: "flex", gap: 10, marginBottom: 32 }}>
        <Button
          onClick={() => setIsAddFormVisible(true)}
          style={{
            borderRadius: 20,
            padding: "4px 20px",
            height: 36,
            borderColor: "#dadce0",
            color: "#1a73e8",
            fontWeight: 500,
            fontSize: 14,
          }}
        >
          Create Guardrail
        </Button>
      </div>

      {/* ── Tab bar ──────────────────────────────────── */}
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

      {/* ── Tab content ──────────────────────────────── */}
      {activeTab === "overview" && (
        <div style={{ display: "flex", gap: 64 }}>
          {/* Left column — overview + details table */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 12px 0" }}>Overview</h2>
            <p style={{ fontSize: 14, color: "#3c4043", lineHeight: 1.7, margin: "0 0 32px 0" }}>
              {card.description}
            </p>

            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 4px 0" }}>Guardrail Details</h2>
            <p style={{ fontSize: 13, color: "#5f6368", margin: "0 0 16px 0" }}>Details are as follows</p>

            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #dadce0" }}>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500, width: 200 }}>Property</th>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500 }}>{card.name}</th>
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

          {/* Right column — metadata sidebar like Vertex */}
          <div style={{ width: 240, flexShrink: 0 }}>
            {/* Guardrail ID */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>Guardrail ID</div>
              <div style={{ fontSize: 13, color: "#202124", wordBreak: "break-all" }}>
                litellm/{card.id}
              </div>
            </div>

            {/* Type */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>Type</div>
              <div style={{ fontSize: 13, color: "#202124" }}>
                {card.category === "litellm" ? "Content Filter" : "Partner"}
              </div>
            </div>

            {/* Tags — pill style like Vertex */}
            {card.tags.length > 0 && (
              <div style={{ marginBottom: 28 }}>
                <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 8 }}>Tags</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {card.tags.map((tag) => (
                    <span
                      key={tag}
                      style={{
                        fontSize: 12,
                        padding: "4px 12px",
                        borderRadius: 16,
                        border: "1px solid #dadce0",
                        color: "#3c4043",
                        backgroundColor: "#fff",
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "eval" && (
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 16px 0" }}>Eval Results</h2>
          <table style={{ width: "100%", maxWidth: 560, borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ backgroundColor: "#f8f9fa", borderBottom: "1px solid #dadce0" }}>
                <th style={{ textAlign: "left", padding: "12px 16px", color: "#5f6368", fontWeight: 500 }}>Metric</th>
                <th style={{ textAlign: "left", padding: "12px 16px", color: "#5f6368", fontWeight: 500 }}>Value</th>
              </tr>
            </thead>
            <tbody>
              {evalRows.map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #f1f3f4" }}>
                  <td style={{ padding: "12px 16px", color: "#3c4043" }}>{row.metric}</td>
                  <td style={{ padding: "12px 16px", color: "#202124", fontWeight: 500 }}>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <AddGuardrailForm
        visible={isAddFormVisible}
        onClose={() => setIsAddFormVisible(false)}
        accessToken={accessToken}
        onSuccess={() => {
          setIsAddFormVisible(false);
          onGuardrailCreated();
        }}
        preset={GUARDRAIL_PRESETS[card.id]}
      />
    </div>
  );
};

export default GuardrailDetailView;
