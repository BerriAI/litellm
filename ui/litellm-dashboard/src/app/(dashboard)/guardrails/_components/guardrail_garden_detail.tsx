import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import AddGuardrailForm from "./add_guardrail_form";
import { Logo } from "@/components/molecules/logo/Logo";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";
import { GuardrailCardInfo } from "./guardrail_garden_data";

interface GuardrailDetailViewProps {
  card: GuardrailCardInfo;
  onBack: () => void;
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailDetailView: React.FC<GuardrailDetailViewProps> = ({ card, onBack, accessToken, onGuardrailCreated }) => {
  const { t } = useTranslation("gateway");
  const [isAddFormVisible, setIsAddFormVisible] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const detailRows = [
    {
      property: t("guardrailsPage.garden.detail.provider"),
      value:
        card.category === "litellm"
          ? t("guardrailsPage.garden.litellmTitle")
          : t("guardrailsPage.garden.detail.partnerGuardrail"),
    },
    ...(card.subcategory ? [{ property: t("guardrailsPage.garden.detail.subcategory"), value: card.subcategory }] : []),
    ...(card.category === "litellm"
      ? [{ property: t("guardrailsPage.garden.detail.cost"), value: t("guardrailsPage.garden.detail.free") }]
      : []),
    ...(card.category === "litellm"
      ? [
          {
            property: t("guardrailsPage.garden.detail.externalDependencies"),
            value: t("guardrailsPage.garden.detail.none"),
          },
        ]
      : []),
    ...(card.category === "litellm"
      ? [{ property: t("guardrailsPage.garden.detail.latency"), value: card.eval?.latency || "<1ms" }]
      : []),
  ];

  const evalRows = card.eval
    ? [
        { metric: t("guardrailsPage.garden.detail.precision"), value: `${card.eval.precision}%` },
        { metric: t("guardrailsPage.garden.detail.recall"), value: `${card.eval.recall}%` },
        { metric: t("guardrailsPage.garden.detail.f1Score"), value: `${card.eval.f1}%` },
        { metric: t("guardrailsPage.garden.detail.testCases"), value: String(card.eval.testCases) },
        { metric: t("guardrailsPage.garden.detail.falsePositives"), value: "0" },
        { metric: t("guardrailsPage.garden.detail.falseNegatives"), value: "0" },
        { metric: t("guardrailsPage.garden.detail.latencyP50"), value: card.eval.latency },
      ]
    : [];

  const tabs = [
    { key: "overview", label: t("guardrailsPage.garden.detail.overview") },
    ...(card.eval ? [{ key: "eval", label: t("guardrailsPage.garden.detail.evalResults") }] : []),
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
        <Logo src={card.logo} label={card.name} className="w-10 h-10 rounded-lg object-contain shrink-0" />
        <h1 style={{ fontSize: 28, fontWeight: 400, color: "#202124", margin: 0, lineHeight: 1.2 }}>{card.name}</h1>
      </div>

      <p style={{ fontSize: 14, color: "#5f6368", margin: "0 0 20px 0", lineHeight: 1.6 }}>{card.description}</p>

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
          {t("guardrailsPage.create.submit")}
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
            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 12px 0" }}>
              {t("guardrailsPage.garden.detail.overview")}
            </h2>
            <p style={{ fontSize: 14, color: "#3c4043", lineHeight: 1.7, margin: "0 0 32px 0" }}>{card.description}</p>

            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 4px 0" }}>
              {t("guardrailsPage.garden.detail.title")}
            </h2>
            <p style={{ fontSize: 13, color: "#5f6368", margin: "0 0 16px 0" }}>
              {t("guardrailsPage.garden.detail.description")}
            </p>

            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #dadce0" }}>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500, width: 200 }}>
                    {t("guardrailsPage.garden.detail.property")}
                  </th>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500 }}>
                    {card.name}
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

          {/* Right column — metadata sidebar like Vertex */}
          <div style={{ width: 240, flexShrink: 0 }}>
            {/* Guardrail ID */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>{t("guardrailsPage.columns.id")}</div>
              <div style={{ fontSize: 13, color: "#202124", wordBreak: "break-all" }}>litellm/{card.id}</div>
            </div>

            {/* Type */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>
                {t("guardrailsPage.garden.detail.type")}
              </div>
              <div style={{ fontSize: 13, color: "#202124" }}>
                {card.category === "litellm"
                  ? t("guardrailsPage.garden.detail.contentFilter")
                  : t("guardrailsPage.garden.detail.partner")}
              </div>
            </div>

            {/* Tags — pill style like Vertex */}
            {card.tags.length > 0 && (
              <div style={{ marginBottom: 28 }}>
                <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 8 }}>
                  {t("guardrailsPage.garden.detail.tags")}
                </div>
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
          <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 16px 0" }}>
            {t("guardrailsPage.garden.detail.evalResults")}
          </h2>
          <table style={{ width: "100%", maxWidth: 560, borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ backgroundColor: "#f8f9fa", borderBottom: "1px solid #dadce0" }}>
                <th style={{ textAlign: "left", padding: "12px 16px", color: "#5f6368", fontWeight: 500 }}>
                  {t("guardrailsPage.garden.detail.metric")}
                </th>
                <th style={{ textAlign: "left", padding: "12px 16px", color: "#5f6368", fontWeight: 500 }}>
                  {t("guardrailsPage.garden.detail.value")}
                </th>
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
        preset={
          GUARDRAIL_PRESETS[card.id] ? { ...GUARDRAIL_PRESETS[card.id], guardrailNameSuggestion: card.name } : undefined
        }
      />
    </div>
  );
};

export default GuardrailDetailView;
