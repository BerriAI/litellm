import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState } from "react";

export interface ContentFilterDetection {
  type: "pattern" | "blocked_word" | "category_keyword";
  pattern_name?: string;
  // matched_text is intentionally excluded to avoid exposing sensitive content
  keyword?: string;
  category?: string;
  severity?: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

interface ContentFilterDetailsProps {
  response: ContentFilterDetection[] | string | null | undefined;
}

const chip = (text: React.ReactNode, tone: "green" | "red" | "blue" | "slate" | "amber" = "slate") => {
  const map: Record<string, string> = {
    green: "bg-success/15 text-success",
    red: "bg-destructive/15 text-destructive",
    blue: "bg-info/10 text-info",
    slate: "bg-muted text-foreground",
    amber: "bg-warning/15 text-warning",
  };
  return <span className={`px-2 py-1 rounded-md text-xs font-medium inline-block ${map[tone]}`}>{text}</span>;
};

interface SectionProps {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children?: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({ title, count, defaultOpen = true, children }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border rounded-lg overflow-hidden">
      <div
        className="flex items-center justify-between p-3 bg-muted cursor-pointer hover:bg-accent"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center">
          <svg
            className={`w-5 h-5 mr-2 transition-transform ${open ? "transform rotate-90" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <h5 className="font-medium">
            {title} {typeof count === "number" && <span className="text-muted-foreground font-normal">({count})</span>}
          </h5>
        </div>
      </div>
      {open && <div className="p-3 border-t bg-card">{children}</div>}
    </div>
  );
};

interface KVProps {
  label: string;
  children?: React.ReactNode;
  mono?: boolean;
}

const KV: React.FC<KVProps> = ({ label, children, mono }) => (
  <div className="flex">
    <span className="font-medium w-1/3">{label}</span>
    <span className={mono ? "font-mono text-sm break-all" : ""}>{children}</span>
  </div>
);

const ContentFilterDetails: React.FC<ContentFilterDetailsProps> = ({ response }) => {
  const { t } = useLanguage();
  // 仅映射已知动作与级别，检测名称、关键词和后端自定义文本保留原样。
  const actionLabel = (action: ContentFilterDetection["action"]) => t(action);
  const severityLabel = (severity: string) => (["high", "medium", "low"].includes(severity) ? t(severity) : severity);
  // Handle case where response is a string (error message) or null
  if (!response || typeof response === "string") {
    if (typeof response === "string" && response) {
      return (
        <div className="bg-card rounded-lg border border-destructive/20 p-4">
          <div className="text-destructive">
            <h5 className="font-medium mb-2">{t("Error")}</h5>
            <p className="text-sm">{response}</p>
          </div>
        </div>
      );
    }
    return null;
  }

  // Ensure response is an array
  const detections: ContentFilterDetection[] = Array.isArray(response) ? response : [];

  if (detections.length === 0) {
    return (
      <div className="bg-card rounded-lg border border-border p-4">
        <div className="text-muted-foreground text-sm">{t("No detections found")}</div>
      </div>
    );
  }

  // Group detections by type
  const patterns = detections.filter((d) => d.type === "pattern");
  const blockedWords = detections.filter((d) => d.type === "blocked_word");
  const categoryKeywords = detections.filter((d) => d.type === "category_keyword");

  // Count actions
  const blockedCount = detections.filter((d) => d.action === "BLOCK").length;
  const maskedCount = detections.filter((d) => d.action === "MASK").length;

  // Summary stats
  const totalDetections = detections.length;

  return (
    <div className="space-y-3">
      {/* Summary Section */}
      <div className="bg-card rounded-lg border border-border p-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <KV label={t("Total Detections:")}>
              <span className="font-semibold">{totalDetections}</span>
            </KV>
            <KV label={t("Actions:")}>
              <div className="flex flex-wrap gap-2">
                {blockedCount > 0 && chip(t("{count} blocked", { count: blockedCount }), "red")}
                {maskedCount > 0 && chip(t("{count} masked", { count: maskedCount }), "blue")}
                {blockedCount === 0 && maskedCount === 0 && chip(t("passed"), "green")}
              </div>
            </KV>
          </div>
          <div className="space-y-2">
            <KV label={t("By Type:")}>
              <div className="flex flex-wrap gap-2">
                {patterns.length > 0 && chip(t("{count} patterns", { count: patterns.length }), "slate")}
                {blockedWords.length > 0 && chip(t("{count} keywords", { count: blockedWords.length }), "slate")}
                {categoryKeywords.length > 0 &&
                  chip(t("{count} categories", { count: categoryKeywords.length }), "slate")}
              </div>
            </KV>
          </div>
        </div>
      </div>

      {/* Patterns Section */}
      {patterns.length > 0 && (
        <Section title={t("Patterns Matched")} count={patterns.length} defaultOpen={true}>
          <div className="space-y-2">
            {patterns.map((detection, idx) => (
              <div key={idx} className="p-3 bg-muted rounded-md">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <KV label={t("Pattern:")}>{detection.pattern_name || t("unknown")}</KV>
                  </div>
                  <div className="space-y-1">
                    <KV label={t("Action:")}>
                      {chip(actionLabel(detection.action), detection.action === "BLOCK" ? "red" : "blue")}
                    </KV>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Blocked Words Section */}
      {blockedWords.length > 0 && (
        <Section title={t("Blocked Words Detected")} count={blockedWords.length} defaultOpen={true}>
          <div className="space-y-2">
            {blockedWords.map((detection, idx) => (
              <div key={idx} className="p-3 bg-muted rounded-md">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <KV label={t("Keyword:")} mono>
                      {detection.keyword || t("unknown")}
                    </KV>
                    {detection.description && <KV label={t("Description:")}>{detection.description}</KV>}
                  </div>
                  <div className="space-y-1">
                    <KV label={t("Action:")}>
                      {chip(actionLabel(detection.action), detection.action === "BLOCK" ? "red" : "blue")}
                    </KV>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Category Keywords Section */}
      {categoryKeywords.length > 0 && (
        <Section title={t("Category Keywords Detected")} count={categoryKeywords.length} defaultOpen={true}>
          <div className="space-y-2">
            {categoryKeywords.map((detection, idx) => (
              <div key={idx} className="p-3 bg-muted rounded-md">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <KV label={t("Category:")}>{detection.category || t("unknown")}</KV>
                    <KV label={t("Keyword:")} mono>
                      {detection.keyword || t("unknown")}
                    </KV>
                    {detection.severity && (
                      <KV label={t("Severity:")}>
                        {chip(
                          severityLabel(detection.severity),
                          detection.severity === "high" ? "red" : detection.severity === "medium" ? "amber" : "slate",
                        )}
                      </KV>
                    )}
                  </div>
                  <div className="space-y-1">
                    <KV label={t("Action:")}>
                      {chip(actionLabel(detection.action), detection.action === "BLOCK" ? "red" : "blue")}
                    </KV>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Raw JSON (for debugging) */}
      <Section title={t("Raw Detection Data")} defaultOpen={false}>
        <pre className="bg-muted rounded-sm p-3 text-xs overflow-x-auto">{JSON.stringify(detections, null, 2)}</pre>
      </Section>
    </div>
  );
};

export default ContentFilterDetails;
