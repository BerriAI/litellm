import React, { useState } from "react";
import { ArrowLeft, Check, Copy, Link2 } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import { useClaudeCodeMarketplaceRequiresKey } from "@/app/(dashboard)/hooks/claudeCodeMarketplace/useClaudeCodeMarketplaceRequiresKey";
import { cn } from "@/lib/cva.config";
import {
  buildMarketplaceSettingsSnippet,
  formatInstallCommand,
  getSourceDisplayText,
  getSourceLink,
  MARKETPLACE_KEY_PLACEHOLDER,
} from "./helpers";
import { Plugin } from "./types";

const SkillSource: React.FC<{ source: Plugin["source"] }> = ({ source }) => {
  const { t } = useTranslation();
  const link = getSourceLink(source);
  const href = link && source.source === "git-subdir" && source.path ? `${link}/tree/main/${source.path}` : link;
  if (href) {
    return (
      <div className="mb-6">
        <div className="mb-1 text-xs text-muted-foreground">{t("claudeCodePlugins.skillDetail.overview.source")}</div>
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 break-all text-[13px] text-info"
        >
          {href.replace("https://", "")}
          <Link2 className="size-3 shrink-0" />
        </a>
      </div>
    );
  }
  if (!source.url) {
    return null;
  }
  return (
    <div className="mb-6">
      <div className="mb-1 text-xs text-muted-foreground">{t("claudeCodePlugins.skillDetail.overview.source")}</div>
      <div className="break-all text-[13px] text-foreground">{getSourceDisplayText(source)}</div>
    </div>
  );
};

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
  const marketplaceRequiresKey = useClaudeCodeMarketplaceRequiresKey();

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const installCommand = formatInstallCommand(skill);

  const settingsSnippet = buildMarketplaceSettingsSnippet(
    typeof window !== "undefined" ? window.location.origin : "<proxy-url>",
    marketplaceRequiresKey,
  );

  const detailRows = [
    ...(skill.category
      ? [{ property: t("claudeCodePlugins.skillDetail.properties.category"), value: skill.category }]
      : []),
    ...(skill.domain ? [{ property: t("claudeCodePlugins.skillDetail.properties.domain"), value: skill.domain }] : []),
    ...(skill.namespace
      ? [{ property: t("claudeCodePlugins.skillDetail.properties.namespace"), value: skill.namespace }]
      : []),
    ...(skill.version
      ? [{ property: t("claudeCodePlugins.skillDetail.properties.version"), value: skill.version }]
      : []),
    ...(skill.author?.name
      ? [{ property: t("claudeCodePlugins.skillDetail.properties.author"), value: skill.author.name }]
      : []),
    ...(skill.created_at
      ? [
          {
            property: t("claudeCodePlugins.skillDetail.properties.added"),
            value: new Date(skill.created_at).toLocaleDateString(),
          },
        ]
      : []),
  ];

  const tabs = [
    { key: "overview", label: t("claudeCodePlugins.skillDetail.tabs.overview") },
    { key: "usage", label: t("claudeCodePlugins.skillDetail.tabs.usage") },
  ];

  return (
    <div className="py-6 pl-0 pr-8">
      {/* Back link */}
      <div
        onClick={onBack}
        className="mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground"
      >
        <ArrowLeft className="size-3" />
        <span>{t("claudeCodePlugins.skillDetail.back")}</span>
      </div>

      {/* Header */}
      <div className="mb-2">
        <h1 className="m-0 text-[28px] font-normal leading-tight text-foreground">{skill.name}</h1>
        {skill.description && (
          <p className="mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground">{skill.description}</p>
        )}
      </div>

      {/* Tab bar */}
      <div className="mb-7 mt-6 border-b border-border">
        <div className="flex">
          {tabs.map((tab) => (
            <div
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                "-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",
                activeTab === tab.key
                  ? "border-info font-medium text-info"
                  : "border-transparent font-normal text-muted-foreground",
              )}
            >
              {tab.label}
            </div>
          ))}
        </div>
      </div>

      {/* Overview tab */}
      {activeTab === "overview" && (
        <div className="flex gap-16">
          {/* Left column */}
          <div className="min-w-0 flex-1">
            <h2 className="m-0 mb-1 text-lg font-normal text-foreground">
              {t("claudeCodePlugins.skillDetail.overview.title")}
            </h2>
            <p className="m-0 mb-4 text-[13px] text-muted-foreground">
              {t("claudeCodePlugins.skillDetail.overview.subtitle")}
            </p>
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="w-40 py-3 text-left font-medium text-muted-foreground">
                    {t("claudeCodePlugins.skillDetail.overview.propertyHeader")}
                  </th>
                  <th className="py-3 text-left font-medium text-muted-foreground">{skill.name}</th>
                </tr>
              </thead>
              <tbody>
                {detailRows.map((row, i) => (
                  <tr key={i} className="border-b border-border">
                    <td className="py-3 text-foreground">{row.property}</td>
                    <td className="py-3 text-foreground">{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Right sidebar */}
          <div className="w-60 shrink-0">
            <div className="mb-6">
              <div className="mb-1 text-xs text-muted-foreground">
                {t("claudeCodePlugins.skillDetail.overview.status")}
              </div>
              <span
                className={cn(
                  "rounded-xl px-2.5 py-[3px] text-xs font-medium",
                  skill.enabled ? "bg-success/10 text-success" : "bg-muted text-muted-foreground",
                )}
              >
                {skill.enabled
                  ? t("claudeCodePlugins.skillDetail.overview.public")
                  : t("claudeCodePlugins.skillDetail.overview.draft")}
              </span>
            </div>

            <SkillSource source={skill.source} />

            {skill.keywords && skill.keywords.length > 0 && (
              <div className="mb-6">
                <div className="mb-2 text-xs text-muted-foreground">
                  {t("claudeCodePlugins.skillDetail.overview.tags")}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {skill.keywords.map((kw) => (
                    <span
                      key={kw}
                      className="rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="mb-1 text-xs text-muted-foreground">
                {t("claudeCodePlugins.skillDetail.overview.skillId")}
              </div>
              <div className="break-all font-mono text-xs text-foreground">{skill.id}</div>
            </div>
          </div>
        </div>
      )}

      {/* How to Use tab */}
      {activeTab === "usage" && (
        <div className="max-w-[640px]">
          <h2 className="m-0 mb-2 text-lg font-normal text-foreground">
            {t("claudeCodePlugins.skillDetail.usage.title")}
          </h2>
          <p className="m-0 mb-6 text-sm leading-relaxed text-muted-foreground">
            {t("claudeCodePlugins.skillDetail.usage.intro")}
          </p>

          {/* Install command */}
          <div className="mb-6 overflow-hidden rounded-lg border border-border">
            <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-2.5">
              <span className="text-[13px] font-medium text-foreground">
                {t("claudeCodePlugins.skillDetail.runInClaudeCode")}
              </span>
              <button
                onClick={() => copyToClipboard(installCommand, "install")}
                className={cn(
                  "flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs",
                  copiedKey === "install" ? "text-success" : "text-info",
                )}
              >
                {copiedKey === "install" ? <Check className="size-3" /> : <Copy className="size-3" />}
                {copiedKey === "install"
                  ? t("claudeCodePlugins.skillDetail.copied")
                  : t("claudeCodePlugins.skillDetail.copy")}
              </button>
            </div>
            <pre className="m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground">{installCommand}</pre>
          </div>

          {/* Shown when the marketplace catalog is stale and the plugin isn't found yet */}
          <div className="mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3">
            <p className="m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground">
              {t("claudeCodePlugins.skillDetail.usage.staleCatalog", { name: skill.name })}
            </p>
            <pre className="m-0 bg-transparent font-mono text-[13px] text-foreground">
              /plugin marketplace update litellm
            </pre>
          </div>

          <p className="m-0 text-[13px] leading-relaxed text-muted-foreground">
            {t("claudeCodePlugins.skillDetail.usage.notConfigured")}{" "}
            <span onClick={() => setActiveTab("setup")} className="cursor-pointer text-info">
              {t("claudeCodePlugins.skillDetail.usage.seeSetup")}
            </span>
          </p>
        </div>
      )}

      {/* Setup tab (linked from usage) */}
      {activeTab === "setup" && (
        <div className="max-w-[640px]">
          <h2 className="m-0 mb-2 text-lg font-normal text-foreground">
            {t("claudeCodePlugins.skillDetail.setup.title")}
          </h2>

          {marketplaceRequiresKey ? (
            <>
              <div className="mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3">
                <p className="m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground">
                  {t("claudeCodePlugins.skillDetail.setup.removeFirst")}
                </p>
                <pre className="m-0 bg-transparent font-mono text-[13px] text-foreground">
                  /plugin marketplace remove litellm
                </pre>
              </div>
              <p className="m-0 mb-3 text-sm leading-relaxed text-muted-foreground">
                <Trans
                  i18nKey="claudeCodePlugins.skillDetail.setup.requiresKey"
                  components={{
                    settings: (
                      <code className="rounded bg-muted px-1.5 py-px text-[13px]">~/.claude/settings.json</code>
                    ),
                    placeholder: (
                      <code className="rounded bg-muted px-1.5 py-px text-[13px]">{MARKETPLACE_KEY_PLACEHOLDER}</code>
                    ),
                  }}
                />
              </p>
            </>
          ) : (
            <>
              {/* Option 1: single command — fastest path for most users */}
              <p className="m-0 mb-3 text-sm leading-relaxed text-muted-foreground">
                {t("claudeCodePlugins.skillDetail.setup.runCommand")}
              </p>
              <div className="mb-6 overflow-hidden rounded-lg border border-border">
                <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-2.5">
                  <span className="text-[13px] font-medium text-foreground">
                    {t("claudeCodePlugins.skillDetail.runInClaudeCode")}
                  </span>
                  <button
                    onClick={() => {
                      const origin = typeof window !== "undefined" ? window.location.origin : "";
                      copyToClipboard(
                        `/plugin marketplace add ${origin}/claude-code/marketplace.json`,
                        "marketplace-cmd",
                      );
                    }}
                    className={cn(
                      "flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs",
                      copiedKey === "marketplace-cmd" ? "text-success" : "text-info",
                    )}
                  >
                    {copiedKey === "marketplace-cmd" ? <Check className="size-3" /> : <Copy className="size-3" />}
                    {copiedKey === "marketplace-cmd"
                      ? t("claudeCodePlugins.skillDetail.copied")
                      : t("claudeCodePlugins.skillDetail.copy")}
                  </button>
                </div>
                <pre className="m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground">
                  {`/plugin marketplace add ${typeof window !== "undefined" ? window.location.origin : "<proxy-url>"}/claude-code/marketplace.json`}
                </pre>
              </div>

              {/* Option 2: settings.json — for persistent config or managed deployments.
                  extraKnownMarketplaces requires source to be a nested object, not a flat string. */}
              <p className="m-0 mb-3 text-sm leading-relaxed text-muted-foreground">
                <Trans
                  i18nKey="claudeCodePlugins.skillDetail.setup.orSettings"
                  components={{
                    settings: (
                      <code className="rounded bg-muted px-1.5 py-px text-[13px]">~/.claude/settings.json</code>
                    ),
                  }}
                />
              </p>
            </>
          )}
          <div className="overflow-hidden rounded-lg border border-border">
            <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-2.5">
              <span className="text-[13px] font-medium text-foreground">~/.claude/settings.json</span>
              <button
                onClick={() => copyToClipboard(settingsSnippet, "settings")}
                className={cn(
                  "flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs",
                  copiedKey === "settings" ? "text-success" : "text-info",
                )}
              >
                {copiedKey === "settings" ? <Check className="size-3" /> : <Copy className="size-3" />}
                {copiedKey === "settings"
                  ? t("claudeCodePlugins.skillDetail.copied")
                  : t("claudeCodePlugins.skillDetail.copy")}
              </button>
            </div>
            <pre className="m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground">{settingsSnippet}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default SkillDetail;
