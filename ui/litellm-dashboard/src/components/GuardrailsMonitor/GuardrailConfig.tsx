import {
  CheckCircleOutlined,
  CodeOutlined,
  PlayCircleOutlined,
  RollbackOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import { Button, Input, Select, Switch } from "antd";
import React, { useState } from "react";

interface GuardrailConfigProps {
  guardrailName: string;
  guardrailType: string;
  provider: string;
}

const versions = [
  { id: "v3", label: "v3 (current)", date: "2026-02-18", author: "admin@company.com", changes: "Adjusted sensitivity for medical terms" },
  { id: "v2", label: "v2", date: "2026-02-10", author: "admin@company.com", changes: "Added custom categories list" },
  { id: "v1", label: "v1", date: "2026-01-28", author: "admin@company.com", changes: "Initial configuration" },
];

export function GuardrailConfig({
  guardrailName,
  guardrailType,
  provider,
}: GuardrailConfigProps) {
  const [action, setAction] = useState("block");
  const [enabled, setEnabled] = useState(true);
  const [customCode, setCustomCode] = useState("");
  const [useCustomCode, setUseCustomCode] = useState(false);
  const [rerunStatus, setRerunStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [version, setVersion] = useState("v3");
  const [showVersionHistory, setShowVersionHistory] = useState(false);

  const handleRerun = () => {
    setRerunStatus("running");
    setTimeout(() => {
      setRerunStatus("success");
      setTimeout(() => setRerunStatus("idle"), 3000);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      {/* Version Bar */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-700">Version:</span>
            <Select
              value={version}
              onChange={setVersion}
              options={versions.map((v) => ({ value: v.id, label: v.label }))}
              style={{ width: 140 }}
            />
            <Button type="link" size="small" onClick={() => setShowVersionHistory(!showVersionHistory)}>
              {showVersionHistory ? "Hide history" : "View history"}
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button icon={<RollbackOutlined />}>Revert</Button>
            <Button type="primary" icon={<SaveOutlined />}>
              Save as v{parseInt(version.replace("v", ""), 10) + 1}
            </Button>
          </div>
        </div>

        {showVersionHistory && (
          <div className="mt-4 border-t border-gray-100 pt-4 space-y-2">
            {versions.map((v) => (
              <div
                key={v.id}
                className={`flex items-center justify-between p-2.5 rounded-md text-sm ${
                  v.id === version ? "bg-blue-50 border border-blue-200" : "bg-gray-50"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className={`font-mono text-xs font-medium ${v.id === version ? "text-blue-600" : "text-gray-500"}`}>
                    {v.id}
                  </span>
                  <span className="text-gray-700">{v.changes}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-gray-500">
                  <span>{v.author}</span>
                  <span>{v.date}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Parameters */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-base font-semibold text-gray-900 mb-1">Parameters</h3>
        <p className="text-xs text-gray-500 mb-5">Configure {guardrailName} behavior</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">Action on Failure</label>
            <Select
              value={action}
              onChange={setAction}
              style={{ width: "100%" }}
              options={[
                { value: "block", label: "Block Request" },
                { value: "flag", label: "Flag for Review" },
                { value: "log", label: "Log Only" },
                { value: "fallback", label: "Use Fallback Response" },
              ]}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">Provider</label>
            <Select
              style={{ width: "100%" }}
              defaultValue={provider}
              options={[
                { value: "bedrock", label: "AWS Bedrock Guardrails" },
                { value: "google", label: "Google Cloud AI Safety" },
                { value: "litellm", label: "LiteLLM Built-in" },
                { value: "custom", label: "Custom Code" },
              ]}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">Guardrail Type</label>
            <Select
              style={{ width: "100%" }}
              defaultValue={guardrailType}
              options={[
                { value: "Content Safety", label: "Content Safety" },
                { value: "PII", label: "PII Detection" },
                { value: "Topic", label: "Topic Restriction" },
                { value: "prompt_injection", label: "Prompt Injection" },
                { value: "custom", label: "Custom" },
              ]}
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1.5">Categories (comma-separated)</label>
            <Input defaultValue="violence, hate_speech, sexual_content, self_harm, illegal_activity" />
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <Switch checked={enabled} onChange={setEnabled} />
            <span className="text-sm text-gray-700">Guardrail enabled in production</span>
          </div>
        </div>
      </div>

      {/* Custom Code Override */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
              <CodeOutlined className="text-gray-500" />
              Custom Code Override
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Replace the built-in guardrail with custom evaluation code
            </p>
          </div>
          <Switch checked={useCustomCode} onChange={setUseCustomCode} />
        </div>

        {useCustomCode && (
          <Input.TextArea
            value={customCode}
            onChange={(e) => setCustomCode(e.target.value)}
            placeholder={`async def evaluate(input_text: str, context: dict) -> dict:
    # Return {"score": 0.0-1.0, "passed": bool, "reason": str}
    # Example:
    if "banned_word" in input_text.lower():
        return {"score": 0.1, "passed": False, "reason": "Banned word detected"}
    return {"score": 0.9, "passed": True, "reason": "No violations"}`}
            rows={10}
            className="font-mono text-sm"
          />
        )}
      </div>

      {/* Re-run on Failing Logs */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-base font-semibold text-gray-900 mb-1">Test Configuration</h3>
        <p className="text-xs text-gray-500 mb-4">
          Re-run this guardrail on recent failing logs to validate your changes
        </p>

        <div className="flex items-center gap-3">
          <Button
            type="primary"
            icon={rerunStatus === "running" ? undefined : <PlayCircleOutlined />}
            loading={rerunStatus === "running"}
            onClick={handleRerun}
          >
            {rerunStatus === "running" ? "Running on 10 samples..." : "Re-run on failing logs"}
          </Button>

          {rerunStatus === "success" && (
            <span className="text-sm text-green-600 flex items-center gap-2">
              <CheckCircleOutlined /> 7/10 would now pass with new config
            </span>
          )}

          {rerunStatus === "error" && (
            <span className="text-sm text-red-600">Error running tests</span>
          )}
        </div>
      </div>
    </div>
  );
}
