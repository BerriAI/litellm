import React, { useState } from "react";
import { Select } from "antd";
import { Button } from "@tremor/react";
import { GuardrailPipeline, PipelineStep, PipelineTestResult } from "./types";
import { testPipelineCall } from "../networking";
import type { CompliancePrompt } from "../../data/compliancePrompts";
import { getFrameworks } from "../../data/compliancePrompts";
import {
  TEST_SOURCE_QUICK,
  TEST_SOURCE_ALL,
  ACTION_LABELS,
  getPromptsForTestSource,
  complianceMatchExpected,
} from "./pipeline_utils";

// ─────────────────────────────────────────────────────────────────────────────
// Style maps
// ─────────────────────────────────────────────────────────────────────────────

const OUTCOME_STYLES: Record<string, { bg: string; color: string; label: string }> = {
  pass: { bg: "#f0fdf4", color: "#16a34a", label: "PASS" },
  fail: { bg: "#fef2f2", color: "#dc2626", label: "FAIL" },
  error: { bg: "#fffbeb", color: "#d97706", label: "ERROR" },
};

const TERMINAL_STYLES: Record<string, { bg: string; color: string }> = {
  allow: { bg: "#f0fdf4", color: "#16a34a" },
  block: { bg: "#fef2f2", color: "#dc2626" },
  modify_response: { bg: "#eff6ff", color: "#2563eb" },
};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface ComplianceRunEntry {
  prompt: CompliancePrompt;
  result: PipelineTestResult | null;
  error?: string;
  matched: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test source options
// ─────────────────────────────────────────────────────────────────────────────

const testSourceOptions = [
  { value: TEST_SOURCE_QUICK, label: "Quick chat (custom message)" },
  ...getFrameworks().map((f) => ({ value: f.name, label: f.name })),
  { value: TEST_SOURCE_ALL, label: "All compliance datasets" },
];

// ─────────────────────────────────────────────────────────────────────────────
// PipelineTestPanel
// ─────────────────────────────────────────────────────────────────────────────

export interface PipelineTestPanelProps {
  pipeline: GuardrailPipeline;
  accessToken: string | null;
  onClose: () => void;
}

export const PipelineTestPanel: React.FC<PipelineTestPanelProps> = ({
  pipeline,
  accessToken,
  onClose,
}) => {
  const [testSource, setTestSource] = useState<string>(TEST_SOURCE_QUICK);
  const [testMessage, setTestMessage] = useState("Hello, can you help me?");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<PipelineTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [complianceResults, setComplianceResults] = useState<ComplianceRunEntry[]>([]);

  const isQuickChat = testSource === TEST_SOURCE_QUICK;
  const promptsForSource = getPromptsForTestSource(testSource);
  const isDataset = promptsForSource.length > 0;

  const handleRunTest = async () => {
    if (!accessToken) return;

    const emptySteps = pipeline.steps.filter((s) => !s.guardrail);
    if (emptySteps.length > 0) {
      setError("All steps must have a guardrail selected");
      return;
    }

    setError(null);
    setIsRunning(true);
    setResult(null);
    setComplianceResults([]);

    if (isQuickChat) {
      try {
        const data = await testPipelineCall(
          accessToken,
          pipeline,
          [{ role: "user", content: testMessage }]
        );
        setResult(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setIsRunning(false);
      }
      return;
    }

    const entries: ComplianceRunEntry[] = [];
    for (const prompt of promptsForSource) {
      try {
        const data = await testPipelineCall(accessToken, pipeline, [
          { role: "user", content: prompt.prompt },
        ]);
        const matched = complianceMatchExpected(prompt.expectedResult, data.terminal_action);
        entries.push({ prompt, result: data, matched });
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : String(e);
        entries.push({
          prompt,
          result: null,
          error: errMsg,
          matched: false,
        });
      }
    }
    setComplianceResults(entries);
    setIsRunning(false);
  };

  return (
    <div
      style={{
        width: 400,
        borderLeft: "1px solid #e5e7eb",
        backgroundColor: "#fff",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
        overflow: "hidden",
      }}
    >
      {/* Panel header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>Test Pipeline</span>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: 18,
            color: "#9ca3af",
            padding: "0 4px",
          }}
        >
          x
        </button>
      </div>

      {/* Input section */}
      <div style={{ padding: 16, borderBottom: "1px solid #e5e7eb" }}>
        <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
          Test with
        </label>
        <Select
          value={testSource}
          onChange={setTestSource}
          options={testSourceOptions}
          style={{ width: "100%", marginBottom: 12 }}
          size="middle"
        />
        {isQuickChat && (
          <>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#6b7280", display: "block", marginBottom: 6 }}>
              Message
            </label>
            <textarea
              value={testMessage}
              onChange={(e) => setTestMessage(e.target.value)}
              placeholder="Enter a test message..."
              rows={3}
              style={{
                width: "100%",
                border: "1px solid #d1d5db",
                borderRadius: 6,
                padding: "8px 10px",
                fontSize: 13,
                resize: "vertical",
                fontFamily: "inherit",
              }}
            />
          </>
        )}
        {isDataset && (
          <div
            style={{
              fontSize: 12,
              color: "#6b7280",
              padding: "8px 10px",
              backgroundColor: "#f9fafb",
              borderRadius: 6,
              marginBottom: 8,
            }}
          >
            {testSource === TEST_SOURCE_ALL
              ? "Run pipeline against all compliance prompts (EU AI Act, GDPR, Topic Blocking, Airline, etc.)."
              : `Run pipeline against ${promptsForSource.length} prompts from "${testSource}".`}
          </div>
        )}
        <Button
          onClick={handleRunTest}
          loading={isRunning}
          style={{ marginTop: 8, width: "100%" }}
        >
          Run Test
        </Button>
      </div>

      {/* Results section */}
      <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
        {error && (
          <div
            style={{
              padding: "10px 12px",
              backgroundColor: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: 6,
              fontSize: 13,
              color: "#dc2626",
              marginBottom: 12,
            }}
          >
            {error}
          </div>
        )}

        {result && (
          <div>
            {/* Step results */}
            {result.step_results.map((step, i) => {
              const style = OUTCOME_STYLES[step.outcome] || OUTCOME_STYLES.error;
              return (
                <div
                  key={i}
                  style={{
                    border: "1px solid #e5e7eb",
                    borderRadius: 8,
                    padding: "10px 12px",
                    marginBottom: 8,
                  }}
                >
                  <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>
                      Step {i + 1}: {step.guardrail_name}
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        backgroundColor: style.bg,
                        color: style.color,
                        padding: "2px 8px",
                        borderRadius: 4,
                      }}
                    >
                      {style.label}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    Action: {ACTION_LABELS[step.action_taken] || step.action_taken}
                    {step.duration_seconds != null && (
                      <span style={{ marginLeft: 8 }}>
                        ({(step.duration_seconds * 1000).toFixed(0)}ms)
                      </span>
                    )}
                  </div>
                  {step.error_detail && (
                    <div style={{ fontSize: 12, color: "#dc2626", marginTop: 4 }}>
                      {step.error_detail}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Terminal result */}
            <div
              style={{
                borderTop: "1px solid #e5e7eb",
                paddingTop: 12,
                marginTop: 4,
              }}
            >
              <div className="flex items-center justify-between">
                <span style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>Result</span>
                {(() => {
                  const ts = TERMINAL_STYLES[result.terminal_action] || TERMINAL_STYLES.block;
                  return (
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        backgroundColor: ts.bg,
                        color: ts.color,
                        padding: "3px 10px",
                        borderRadius: 4,
                        textTransform: "uppercase",
                      }}
                    >
                      {result.terminal_action === "modify_response" ? "Custom Response" : result.terminal_action}
                    </span>
                  );
                })()}
              </div>
              {result.error_message && (
                <div style={{ fontSize: 12, color: "#dc2626", marginTop: 6 }}>
                  {result.error_message}
                </div>
              )}
              {result.modify_response_message && (
                <div style={{ fontSize: 12, color: "#2563eb", marginTop: 6 }}>
                  Response: {result.modify_response_message}
                </div>
              )}
            </div>
          </div>
        )}

        {complianceResults.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: "#111827",
                marginBottom: 8,
              }}
            >
              Compliance dataset
            </div>
            <div
              style={{
                fontSize: 12,
                color: "#6b7280",
                marginBottom: 10,
              }}
            >
              {complianceResults.filter((e) => e.matched).length} / {complianceResults.length} matched
              expected
            </div>
            <div
              style={{
                maxHeight: 320,
                overflowY: "auto",
                border: "1px solid #e5e7eb",
                borderRadius: 8,
              }}
            >
              {complianceResults.map((entry, i) => {
                const actual =
                  entry.result?.terminal_action ?? (entry.error ? "error" : "—");
                const matchStyle = entry.matched
                  ? { bg: "#f0fdf4", color: "#16a34a" }
                  : { bg: "#fef2f2", color: "#dc2626" };
                return (
                  <div
                    key={entry.prompt.id ?? i}
                    style={{
                      padding: "8px 10px",
                      borderBottom:
                        i < complianceResults.length - 1
                          ? "1px solid #e5e7eb"
                          : "none",
                      fontSize: 12,
                    }}
                  >
                    <div
                      style={{
                        color: "#374151",
                        marginBottom: 4,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                      title={entry.prompt.prompt}
                    >
                      {entry.prompt.prompt}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        flexWrap: "wrap",
                      }}
                    >
                      <span style={{ color: "#6b7280" }}>
                        expected: {entry.prompt.expectedResult}
                      </span>
                      <span style={{ color: "#9ca3af" }}>→</span>
                      <span style={{ color: "#6b7280" }}>
                        actual: {actual}
                      </span>
                      <span
                        style={{
                          backgroundColor: matchStyle.bg,
                          color: matchStyle.color,
                          padding: "1px 6px",
                          borderRadius: 4,
                          fontWeight: 600,
                        }}
                      >
                        {entry.matched ? "✓" : "✗"}
                      </span>
                    </div>
                    {entry.error && (
                      <div style={{ color: "#dc2626", marginTop: 4 }}>
                        {entry.error}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {!result && !error && complianceResults.length === 0 && (
          <div style={{ textAlign: "center", color: "#9ca3af", fontSize: 13, marginTop: 24 }}>
            Choose a test source above (quick chat or a compliance dataset) and click &quot;Run Test&quot;
          </div>
        )}
      </div>
    </div>
  );
};
