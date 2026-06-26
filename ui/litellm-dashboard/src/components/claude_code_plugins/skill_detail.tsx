import React, { useState } from "react";
import { ArrowLeftOutlined, CopyOutlined, CheckOutlined, LinkOutlined } from "@ant-design/icons";
import { useTranslation, Trans } from "react-i18next";
import { formatInstallCommand } from "./helpers";
import { Plugin } from "./types";

interface SkillDetailProps {
  skill: Plugin;
  onBack: () => void;
  isAdmin?: boolean;
  accessToken?: string | null;
  onPublishClick?: () => void;
}

const SkillDetail: React.FC<SkillDetailProps> = ({ skill, onBack }) => {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState("overview");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const sourceUrl = (() => {
    const src = skill.source;
    if (src.source === "github" && src.repo) return `https://github.com/${src.repo}`;
    if (src.source === "git-subdir" && src.url) return src.path ? `${src.url}/tree/main/${src.path}` : src.url;
    if (src.source === "url" && src.url) return src.url;
    return null;
  })();

  const installCommand = formatInstallCommand(skill);

  const detailRows = [
    ...(skill.category
      ? [{ property: t("claudeCodePluginsPage.skillDetail.propCategory"), value: skill.category }]
      : []),
    ...(skill.domain ? [{ property: t("claudeCodePluginsPage.skillDetail.propDomain"), value: skill.domain }] : []),
    ...(skill.namespace
      ? [{ property: t("claudeCodePluginsPage.skillDetail.propNamespace"), value: skill.namespace }]
      : []),
    ...(skill.version ? [{ property: t("claudeCodePluginsPage.skillDetail.propVersion"), value: skill.version }] : []),
    ...(skill.author?.name
      ? [{ property: t("claudeCodePluginsPage.skillDetail.propAuthor"), value: skill.author.name }]
      : []),
    ...(skill.created_at
      ? [
          {
            property: t("claudeCodePluginsPage.skillDetail.propAdded"),
            value: new Date(skill.created_at).toLocaleDateString(),
          },
        ]
      : []),
  ];

  const tabs = [
    { key: "overview", label: t("claudeCodePluginsPage.skillDetail.tabOverview") },
    { key: "usage", label: t("claudeCodePluginsPage.skillDetail.tabUsage") },
  ];

  return (
    <div style={{ padding: "24px 32px 24px 0" }}>
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
        <span>{t("claudeCodePluginsPage.skillDetail.backLink")}</span>
      </div>

      {/* Header */}
      <div style={{ marginBottom: 8 }}>
        <h1 style={{ fontSize: 28, fontWeight: 400, color: "#202124", margin: 0, lineHeight: 1.2 }}>{skill.name}</h1>
        {skill.description && (
          <p style={{ fontSize: 14, color: "#5f6368", margin: "8px 0 0 0", lineHeight: 1.6 }}>{skill.description}</p>
        )}
      </div>

      {/* Tab bar */}
      <div style={{ borderBottom: "1px solid #dadce0", marginBottom: 28, marginTop: 24 }}>
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

      {/* Overview tab */}
      {activeTab === "overview" && (
        <div style={{ display: "flex", gap: 64 }}>
          {/* Left column */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 4px 0" }}>
              {t("claudeCodePluginsPage.skillDetail.skillDetailsHeading")}
            </h2>
            <p style={{ fontSize: 13, color: "#5f6368", margin: "0 0 16px 0" }}>
              {t("claudeCodePluginsPage.skillDetail.skillDetailsSubtitle")}
            </p>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #dadce0" }}>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500, width: 160 }}>
                    {t("claudeCodePluginsPage.skillDetail.tableColProperty")}
                  </th>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500 }}>
                    {skill.name}
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

          {/* Right sidebar */}
          <div style={{ width: 240, flexShrink: 0 }}>
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>
                {t("claudeCodePluginsPage.skillDetail.sidebarStatus")}
              </div>
              <span
                style={{
                  fontSize: 12,
                  padding: "3px 10px",
                  borderRadius: 12,
                  backgroundColor: skill.enabled ? "#e6f4ea" : "#f1f3f4",
                  color: skill.enabled ? "#137333" : "#5f6368",
                  fontWeight: 500,
                }}
              >
                {skill.enabled
                  ? t("claudeCodePluginsPage.skillDetail.statusPublic")
                  : t("claudeCodePluginsPage.skillDetail.statusDraft")}
              </span>
            </div>

            {sourceUrl && (
              <div style={{ marginBottom: 24 }}>
                <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>
                  {t("claudeCodePluginsPage.skillDetail.sidebarSource")}
                </div>
                <a
                  href={sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    fontSize: 13,
                    color: "#1a73e8",
                    wordBreak: "break-all",
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                  }}
                >
                  {sourceUrl.replace("https://", "")}
                  <LinkOutlined style={{ fontSize: 11, flexShrink: 0 }} />
                </a>
              </div>
            )}

            {skill.keywords && skill.keywords.length > 0 && (
              <div style={{ marginBottom: 24 }}>
                <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 8 }}>
                  {t("claudeCodePluginsPage.skillDetail.sidebarTags")}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {skill.keywords.map((kw) => (
                    <span
                      key={kw}
                      style={{
                        fontSize: 12,
                        padding: "4px 12px",
                        borderRadius: 16,
                        border: "1px solid #dadce0",
                        color: "#3c4043",
                        backgroundColor: "#fff",
                      }}
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>
                {t("claudeCodePluginsPage.skillDetail.sidebarSkillId")}
              </div>
              <div style={{ fontSize: 12, fontFamily: "monospace", color: "#3c4043", wordBreak: "break-all" }}>
                {skill.id}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* How to Use tab */}
      {activeTab === "usage" && (
        <div style={{ maxWidth: 640 }}>
          <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 8px 0" }}>
            {t("claudeCodePluginsPage.skillDetail.usingThisSkill")}
          </h2>
          <p style={{ fontSize: 14, color: "#5f6368", margin: "0 0 24px 0", lineHeight: 1.6 }}>
            {t("claudeCodePluginsPage.skillDetail.usageInstruction")}
          </p>

          {/* Install command */}
          <div
            style={{
              border: "1px solid #dadce0",
              borderRadius: 8,
              overflow: "hidden",
              marginBottom: 24,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "10px 16px",
                backgroundColor: "#f8f9fa",
                borderBottom: "1px solid #dadce0",
              }}
            >
              <span style={{ fontSize: 13, color: "#3c4043", fontWeight: 500 }}>
                {t("claudeCodePluginsPage.skillDetail.runInClaudeCode")}
              </span>
              <button
                onClick={() => copyToClipboard(installCommand, "install")}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  fontSize: 12,
                  color: copiedKey === "install" ? "#137333" : "#1a73e8",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: 0,
                }}
              >
                {copiedKey === "install" ? <CheckOutlined /> : <CopyOutlined />}
                {copiedKey === "install" ? t("common.copied") : t("common.copy")}
              </button>
            </div>
            <pre
              style={{
                margin: 0,
                padding: "14px 16px",
                fontSize: 14,
                fontFamily: "monospace",
                color: "#202124",
                backgroundColor: "#fff",
              }}
            >
              {installCommand}
            </pre>
          </div>

          <p style={{ fontSize: 13, color: "#5f6368", lineHeight: 1.6, margin: 0 }}>
            {t("claudeCodePluginsPage.skillDetail.noMarketplace")}{" "}
            <span onClick={() => setActiveTab("setup")} style={{ color: "#1a73e8", cursor: "pointer" }}>
              {t("claudeCodePluginsPage.skillDetail.seeSetup")}
            </span>
          </p>
        </div>
      )}

      {/* Setup tab (linked from usage) */}
      {activeTab === "setup" && (
        <div style={{ maxWidth: 640 }}>
          <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 8px 0" }}>
            {t("claudeCodePluginsPage.skillDetail.setupHeading")}
          </h2>
          <p style={{ fontSize: 14, color: "#5f6368", margin: "0 0 24px 0", lineHeight: 1.6 }}>
            <Trans
              i18nKey="claudeCodePluginsPage.skillDetail.setupInstruction"
              components={{
                code: (
                  <code style={{ fontSize: 13, backgroundColor: "#f1f3f4", padding: "1px 6px", borderRadius: 4 }} />
                ),
              }}
            />
          </p>
          <div
            style={{
              border: "1px solid #dadce0",
              borderRadius: 8,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "10px 16px",
                backgroundColor: "#f8f9fa",
                borderBottom: "1px solid #dadce0",
              }}
            >
              <span style={{ fontSize: 13, color: "#3c4043", fontWeight: 500 }}>~/.claude/settings.json</span>
              <button
                onClick={() => {
                  const snippet = JSON.stringify(
                    {
                      extraKnownMarketplaces: {
                        "my-org": {
                          source: "url",
                          url: `${typeof window !== "undefined" ? window.location.origin : ""}/claude-code/marketplace.json`,
                        },
                      },
                    },
                    null,
                    2,
                  );
                  copyToClipboard(snippet, "settings");
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  fontSize: 12,
                  color: copiedKey === "settings" ? "#137333" : "#1a73e8",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: 0,
                }}
              >
                {copiedKey === "settings" ? <CheckOutlined /> : <CopyOutlined />}
                {copiedKey === "settings" ? t("common.copied") : t("common.copy")}
              </button>
            </div>
            <pre
              style={{
                margin: 0,
                padding: "14px 16px",
                fontSize: 13,
                fontFamily: "monospace",
                color: "#202124",
                backgroundColor: "#fff",
              }}
            >
              {JSON.stringify(
                {
                  extraKnownMarketplaces: {
                    "my-org": {
                      source: "url",
                      url: `${typeof window !== "undefined" ? window.location.origin : "<proxy-url>"}/claude-code/marketplace.json`,
                    },
                  },
                },
                null,
                2,
              )}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default SkillDetail;
