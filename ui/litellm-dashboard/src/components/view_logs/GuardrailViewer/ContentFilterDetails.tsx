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
    green: "bg-green-100 text-green-800",
    red: "bg-red-100 text-red-800",
    blue: "bg-blue-50 text-blue-700",
    slate: "bg-slate-100 text-slate-800",
    amber: "bg-amber-100 text-amber-800",
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
        className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
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
            {title} {typeof count === "number" && <span className="text-gray-500 font-normal">({count})</span>}
          </h5>
        </div>
      </div>
      {open && <div className="p-3 border-t bg-white">{children}</div>}
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
  // Handle case where response is a string (error message) or null
  if (!response || typeof response === "string") {
    if (typeof response === "string" && response) {
      return (
        <div className="bg-white rounded-lg border border-red-200 p-4">
          <div className="text-red-800">
            <h5 className="font-medium mb-2">Error</h5>
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
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="text-gray-600 text-sm">No detections found</div>
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
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <KV label="Total Detections:">
              <span className="font-semibold">{totalDetections}</span>
            </KV>
            <KV label="Actions:">
              <div className="flex flex-wrap gap-2">
                {blockedCount > 0 && chip(`${blockedCount} blocked`, "red")}
                {maskedCount > 0 && chip(`${maskedCount} masked`, "blue")}
                {blockedCount === 0 && maskedCount === 0 && chip("passed", "green")}
              </div>
            </KV>
          </div>
          <div className="space-y-2">
            <KV label="By Type:">
              <div className="flex flex-wrap gap-2">
                {patterns.length > 0 && chip(`${patterns.length} patterns`, "slate")}
                {blockedWords.length > 0 && chip(`${blockedWords.length} keywords`, "slate")}
                {categoryKeywords.length > 0 && chip(`${categoryKeywords.length} categories`, "slate")}
              </div>
            </KV>
          </div>
        </div>
      </div>

      {/* Patterns Section */}
      {patterns.length > 0 && (
        <Section title="Patterns Matched" count={patterns.length} defaultOpen={true}>
          <div className="space-y-2">
            {patterns.map((detection, idx) => (
              <div key={idx} className="p-3 bg-gray-50 rounded-md">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <KV label="Pattern:">{detection.pattern_name || "unknown"}</KV>
                  </div>
                  <div className="space-y-1">
                    <KV label="Action:">{chip(detection.action, detection.action === "BLOCK" ? "red" : "blue")}</KV>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Blocked Words Section */}
      {blockedWords.length > 0 && (
        <Section title="Blocked Words Detected" count={blockedWords.length} defaultOpen={true}>
          <div className="space-y-2">
            {blockedWords.map((detection, idx) => (
              <div key={idx} className="p-3 bg-gray-50 rounded-md">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <KV label="Keyword:" mono>
                      {detection.keyword || "unknown"}
                    </KV>
                    {detection.description && <KV label="Description:">{detection.description}</KV>}
                  </div>
                  <div className="space-y-1">
                    <KV label="Action:">{chip(detection.action, detection.action === "BLOCK" ? "red" : "blue")}</KV>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Category Keywords Section */}
      {categoryKeywords.length > 0 && (
        <Section title="Category Keywords Detected" count={categoryKeywords.length} defaultOpen={true}>
          <div className="space-y-2">
            {categoryKeywords.map((detection, idx) => (
              <div key={idx} className="p-3 bg-gray-50 rounded-md">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <KV label="Category:">{detection.category || "unknown"}</KV>
                    <KV label="Keyword:" mono>
                      {detection.keyword || "unknown"}
                    </KV>
                    {detection.severity && (
                      <KV label="Severity:">
                        {chip(detection.severity, detection.severity === "high" ? "red" : detection.severity === "medium" ? "amber" : "slate")}
                      </KV>
                    )}
                  </div>
                  <div className="space-y-1">
                    <KV label="Action:">{chip(detection.action, detection.action === "BLOCK" ? "red" : "blue")}</KV>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Raw JSON (for debugging) */}
      <Section title="Raw Detection Data" defaultOpen={false}>
        <pre className="bg-gray-50 rounded p-3 text-xs overflow-x-auto">{JSON.stringify(detections, null, 2)}</pre>
      </Section>
    </div>
  );
};

export default ContentFilterDetails;

