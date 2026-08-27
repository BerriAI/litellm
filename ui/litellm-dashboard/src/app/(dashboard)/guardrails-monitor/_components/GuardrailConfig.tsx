import { CircleCheck, CirclePlay, Code, Save, Undo2 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import React, { useId, useState } from "react";

interface GuardrailConfigProps {
  guardrailName: string;
  guardrailType: string;
  provider: string;
}

const versions = [
  {
    id: "v3",
    label: "v3 (current)",
    date: "2026-02-18",
    author: "admin@company.com",
    changes: "Adjusted sensitivity for medical terms",
  },
  { id: "v2", label: "v2", date: "2026-02-10", author: "admin@company.com", changes: "Added custom categories list" },
  { id: "v1", label: "v1", date: "2026-01-28", author: "admin@company.com", changes: "Initial configuration" },
];

const ACTION_ITEMS = [
  { value: "block", label: "Block Request" },
  { value: "flag", label: "Flag for Review" },
  { value: "log", label: "Log Only" },
  { value: "fallback", label: "Use Fallback Response" },
];

const PROVIDER_ITEMS = [
  { value: "bedrock", label: "AWS Bedrock Guardrails" },
  { value: "google", label: "Google Cloud AI Safety" },
  { value: "litellm", label: "LiteLLM Built-in" },
  { value: "custom", label: "Custom Code" },
];

const GUARDRAIL_TYPE_ITEMS = [
  { value: "Content Safety", label: "Content Safety" },
  { value: "PII", label: "PII Detection" },
  { value: "Topic", label: "Topic Restriction" },
  { value: "prompt_injection", label: "Prompt Injection" },
  { value: "custom", label: "Custom" },
];

export function GuardrailConfig({ guardrailName, guardrailType, provider }: GuardrailConfigProps) {
  const [action, setAction] = useState("block");
  const [enabled, setEnabled] = useState(true);
  const [customCode, setCustomCode] = useState("");
  const [useCustomCode, setUseCustomCode] = useState(false);
  const [rerunStatus, setRerunStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [version, setVersion] = useState("v3");
  const [showVersionHistory, setShowVersionHistory] = useState(false);
  const enabledToggleId = useId();

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
      <div className="bg-card border border-border rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-foreground">Version:</span>
            <Select
              items={versions.map((v) => ({ value: v.id, label: v.label }))}
              value={version}
              onValueChange={(value: string | null) => value && setVersion(value)}
            >
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {versions.map((v) => (
                  <SelectItem key={v.id} value={v.id}>
                    {v.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="link" size="sm" onClick={() => setShowVersionHistory(!showVersionHistory)}>
              {showVersionHistory ? "Hide history" : "View history"}
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline">
              <Undo2 />
              Revert
            </Button>
            <Button>
              <Save />
              Save as v{parseInt(version.replace("v", ""), 10) + 1}
            </Button>
          </div>
        </div>

        {showVersionHistory && (
          <div className="mt-4 border-t border-border pt-4 space-y-2">
            {versions.map((v) => (
              <div
                key={v.id}
                className={`flex items-center justify-between p-2.5 rounded-md text-sm ${
                  v.id === version ? "bg-info/10 border border-info/20" : "bg-muted"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`font-mono text-xs font-medium ${v.id === version ? "text-info" : "text-muted-foreground"}`}
                  >
                    {v.id}
                  </span>
                  <span className="text-foreground">{v.changes}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>{v.author}</span>
                  <span>{v.date}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Parameters */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-base font-semibold text-foreground mb-1">Parameters</h3>
        <p className="text-xs text-muted-foreground mb-5">Configure {guardrailName} behavior</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Action on Failure</label>
            <Select
              items={ACTION_ITEMS}
              value={action}
              onValueChange={(value: string | null) => value && setAction(value)}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ACTION_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Provider</label>
            <Select items={PROVIDER_ITEMS} defaultValue={provider}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PROVIDER_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Guardrail Type</label>
            <Select items={GUARDRAIL_TYPE_ITEMS} defaultValue={guardrailType}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {GUARDRAIL_TYPE_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-foreground mb-1.5">Categories (comma-separated)</label>
            <Input defaultValue="violence, hate_speech, sexual_content, self_harm, illegal_activity" />
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <Switch id={enabledToggleId} checked={enabled} onCheckedChange={setEnabled} />
            <Label htmlFor={enabledToggleId} className="font-normal text-foreground">
              Guardrail enabled in production
            </Label>
          </div>
        </div>
      </div>

      {/* Custom Code Override */}
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
              <Code className="size-4 text-muted-foreground" />
              Custom Code Override
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Replace the built-in guardrail with custom evaluation code
            </p>
          </div>
          <Switch aria-label="Custom Code Override" checked={useCustomCode} onCheckedChange={setUseCustomCode} />
        </div>

        {useCustomCode && (
          <Textarea
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
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-base font-semibold text-foreground mb-1">Test Configuration</h3>
        <p className="text-xs text-muted-foreground mb-4">
          Re-run this guardrail on recent failing logs to validate your changes
        </p>

        <div className="flex items-center gap-3">
          <Button disabled={rerunStatus === "running"} aria-busy={rerunStatus === "running"} onClick={handleRerun}>
            {rerunStatus === "running" ? null : <CirclePlay />}
            {rerunStatus === "running" ? "Running on 10 samples..." : "Re-run on failing logs"}
          </Button>

          {rerunStatus === "success" && (
            <span className="text-sm text-success flex items-center gap-2">
              <CircleCheck className="size-4" /> 7/10 would now pass with new config
            </span>
          )}

          {rerunStatus === "error" && <span className="text-sm text-destructive">Error running tests</span>}
        </div>
      </div>
    </div>
  );
}
