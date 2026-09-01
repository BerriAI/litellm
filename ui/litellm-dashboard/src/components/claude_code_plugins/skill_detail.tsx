import React, { useState } from "react";
import { ArrowLeft, Check, Copy, Link2 } from "lucide-react";
import { cn } from "@/lib/cva.config";
import { buildMarketplaceSettingsSnippet, formatInstallCommand } from "./helpers";
import { Plugin } from "./types";

interface SkillDetailProps {
  skill: Plugin;
  onBack: () => void;
  isAdmin?: boolean;
  accessToken?: string | null;
  onPublishClick?: () => void;
}

const SkillDetail: React.FC<SkillDetailProps> = ({ skill, onBack }) => {
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

  const settingsSnippet = buildMarketplaceSettingsSnippet(
    typeof window !== "undefined" ? window.location.origin : "<proxy-url>",
  );

  const detailRows = [
    ...(skill.category ? [{ property: "Category", value: skill.category }] : []),
    ...(skill.domain ? [{ property: "Domain", value: skill.domain }] : []),
    ...(skill.namespace ? [{ property: "Namespace", value: skill.namespace }] : []),
    ...(skill.version ? [{ property: "Version", value: skill.version }] : []),
    ...(skill.author?.name ? [{ property: "Author", value: skill.author.name }] : []),
    ...(skill.created_at ? [{ property: "Added", value: new Date(skill.created_at).toLocaleDateString() }] : []),
  ];

  const tabs = [
    { key: "overview", label: "Overview" },
    { key: "usage", label: "How to Use" },
  ];

  return (
    <div className="py-6 pl-0 pr-8">
      {/* Back link */}
      <div
        onClick={onBack}
        className="mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground"
      >
        <ArrowLeft className="size-3" />
        <span>Skills</span>
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
            <h2 className="m-0 mb-1 text-lg font-normal text-foreground">Skill Details</h2>
            <p className="m-0 mb-4 text-[13px] text-muted-foreground">Metadata registered with this skill</p>
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="w-40 py-3 text-left font-medium text-muted-foreground">Property</th>
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
              <div className="mb-1 text-xs text-muted-foreground">Status</div>
              <span
                className={cn(
                  "rounded-xl px-2.5 py-[3px] text-xs font-medium",
                  skill.enabled ? "bg-success/10 text-success" : "bg-muted text-muted-foreground",
                )}
              >
                {skill.enabled ? "Public" : "Draft"}
              </span>
            </div>

            {sourceUrl && (
              <div className="mb-6">
                <div className="mb-1 text-xs text-muted-foreground">Source</div>
                <a
                  href={sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 break-all text-[13px] text-info"
                >
                  {sourceUrl.replace("https://", "")}
                  <Link2 className="size-3 shrink-0" />
                </a>
              </div>
            )}

            {skill.keywords && skill.keywords.length > 0 && (
              <div className="mb-6">
                <div className="mb-2 text-xs text-muted-foreground">Tags</div>
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
              <div className="mb-1 text-xs text-muted-foreground">Skill ID</div>
              <div className="break-all font-mono text-xs text-foreground">{skill.id}</div>
            </div>
          </div>
        </div>
      )}

      {/* How to Use tab */}
      {activeTab === "usage" && (
        <div className="max-w-[640px]">
          <h2 className="m-0 mb-2 text-lg font-normal text-foreground">Using this skill</h2>
          <p className="m-0 mb-6 text-sm leading-relaxed text-muted-foreground">
            Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:
          </p>

          {/* Install command */}
          <div className="mb-6 overflow-hidden rounded-lg border border-border">
            <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-2.5">
              <span className="text-[13px] font-medium text-foreground">Run in Claude Code</span>
              <button
                onClick={() => copyToClipboard(installCommand, "install")}
                className={cn(
                  "flex cursor-pointer items-center gap-1 border-none bg-none p-0 text-xs",
                  copiedKey === "install" ? "text-success" : "text-info",
                )}
              >
                {copiedKey === "install" ? <Check className="size-3" /> : <Copy className="size-3" />}
                {copiedKey === "install" ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground">{installCommand}</pre>
          </div>

          {/* Shown when the marketplace catalog is stale and the plugin isn't found yet */}
          <div className="mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3">
            <p className="m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground">
              If you see &quot;Plugin {skill.name} not found in marketplace&quot;, update the catalog first:
            </p>
            <pre className="m-0 bg-transparent font-mono text-[13px] text-foreground">
              /plugin marketplace update litellm
            </pre>
          </div>

          <p className="m-0 text-[13px] leading-relaxed text-muted-foreground">
            Don&apos;t have the marketplace configured yet?{" "}
            <span onClick={() => setActiveTab("setup")} className="cursor-pointer text-info">
              See one-time setup →
            </span>
          </p>
        </div>
      )}

      {/* Setup tab (linked from usage) */}
      {activeTab === "setup" && (
        <div className="max-w-[640px]">
          <h2 className="m-0 mb-2 text-lg font-normal text-foreground">One-time marketplace setup</h2>

          {/* Option 1: single command — fastest path for most users */}
          <p className="m-0 mb-3 text-sm leading-relaxed text-muted-foreground">
            Run this command in Claude Code to register the marketplace:
          </p>
          <div className="mb-6 overflow-hidden rounded-lg border border-border">
            <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-2.5">
              <span className="text-[13px] font-medium text-foreground">Run in Claude Code</span>
              <button
                onClick={() => {
                  const origin = typeof window !== "undefined" ? window.location.origin : "";
                  copyToClipboard(`/plugin marketplace add ${origin}/claude-code/marketplace.json`, "marketplace-cmd");
                }}
                className={cn(
                  "flex cursor-pointer items-center gap-1 border-none bg-none p-0 text-xs",
                  copiedKey === "marketplace-cmd" ? "text-success" : "text-info",
                )}
              >
                {copiedKey === "marketplace-cmd" ? <Check className="size-3" /> : <Copy className="size-3" />}
                {copiedKey === "marketplace-cmd" ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground">
              {`/plugin marketplace add ${typeof window !== "undefined" ? window.location.origin : "<proxy-url>"}/claude-code/marketplace.json`}
            </pre>
          </div>

          {/* Option 2: settings.json — for persistent config or managed deployments.
              extraKnownMarketplaces requires source to be a nested object, not a flat string. */}
          <p className="m-0 mb-3 text-sm leading-relaxed text-muted-foreground">
            Or add this to <code className="rounded bg-muted px-1.5 py-px text-[13px]">~/.claude/settings.json</code>{" "}
            for a persistent configuration:
          </p>
          <div className="overflow-hidden rounded-lg border border-border">
            <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-2.5">
              <span className="text-[13px] font-medium text-foreground">~/.claude/settings.json</span>
              <button
                onClick={() => copyToClipboard(settingsSnippet, "settings")}
                className={cn(
                  "flex cursor-pointer items-center gap-1 border-none bg-none p-0 text-xs",
                  copiedKey === "settings" ? "text-success" : "text-info",
                )}
              >
                {copiedKey === "settings" ? <Check className="size-3" /> : <Copy className="size-3" />}
                {copiedKey === "settings" ? "Copied" : "Copy"}
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
