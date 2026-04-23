import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  CheckCircle,
  Code,
  PlayCircle,
  RotateCcw,
  Save,
} from "lucide-react";
import React, { useState } from "react";

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
  {
    id: "v2",
    label: "v2",
    date: "2026-02-10",
    author: "admin@company.com",
    changes: "Added custom categories list",
  },
  {
    id: "v1",
    label: "v1",
    date: "2026-01-28",
    author: "admin@company.com",
    changes: "Initial configuration",
  },
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
  const [rerunStatus, setRerunStatus] = useState<
    "idle" | "running" | "success" | "error"
  >("idle");
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
      <div className="bg-background border border-border rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-foreground">Version:</span>
            <Select value={version} onValueChange={setVersion}>
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
            <Button
              variant="link"
              size="sm"
              onClick={() => setShowVersionHistory(!showVersionHistory)}
            >
              {showVersionHistory ? "Hide history" : "View history"}
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline">
              <RotateCcw className="h-4 w-4" />
              Revert
            </Button>
            <Button>
              <Save className="h-4 w-4" />
              Save as v{parseInt(version.replace("v", ""), 10) + 1}
            </Button>
          </div>
        </div>

        {showVersionHistory && (
          <div className="mt-4 border-t border-border pt-4 space-y-2">
            {versions.map((v) => (
              <div
                key={v.id}
                className={cn(
                  "flex items-center justify-between p-2.5 rounded-md text-sm",
                  v.id === version
                    ? "bg-accent border border-border"
                    : "bg-muted",
                )}
              >
                <div className="flex items-center gap-3">
                  <span
                    className={cn(
                      "font-mono text-xs font-medium",
                      v.id === version
                        ? "text-primary"
                        : "text-muted-foreground",
                    )}
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
      <div className="bg-background border border-border rounded-lg p-6">
        <h3 className="text-base font-semibold text-foreground mb-1">
          Parameters
        </h3>
        <p className="text-xs text-muted-foreground mb-5">
          Configure {guardrailName} behavior
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <Label className="block mb-1.5">Action on Failure</Label>
            <Select value={action} onValueChange={setAction}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="block">Block Request</SelectItem>
                <SelectItem value="flag">Flag for Review</SelectItem>
                <SelectItem value="log">Log Only</SelectItem>
                <SelectItem value="fallback">Use Fallback Response</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label className="block mb-1.5">Provider</Label>
            <Select defaultValue={provider}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bedrock">AWS Bedrock Guardrails</SelectItem>
                <SelectItem value="google">Google Cloud AI Safety</SelectItem>
                <SelectItem value="litellm">LiteLLM Built-in</SelectItem>
                <SelectItem value="custom">Custom Code</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label className="block mb-1.5">Guardrail Type</Label>
            <Select defaultValue={guardrailType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Content Safety">Content Safety</SelectItem>
                <SelectItem value="PII">PII Detection</SelectItem>
                <SelectItem value="Topic">Topic Restriction</SelectItem>
                <SelectItem value="prompt_injection">Prompt Injection</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="md:col-span-2">
            <Label className="block mb-1.5">
              Categories (comma-separated)
            </Label>
            <Input defaultValue="violence, hate_speech, sexual_content, self_harm, illegal_activity" />
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <Switch checked={enabled} onCheckedChange={setEnabled} />
            <span className="text-sm text-foreground">
              Guardrail enabled in production
            </span>
          </div>
        </div>
      </div>

      {/* Custom Code Override */}
      <div className="bg-background border border-border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
              <Code className="h-4 w-4 text-muted-foreground" />
              Custom Code Override
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Replace the built-in guardrail with custom evaluation code
            </p>
          </div>
          <Switch checked={useCustomCode} onCheckedChange={setUseCustomCode} />
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
      <div className="bg-background border border-border rounded-lg p-6">
        <h3 className="text-base font-semibold text-foreground mb-1">
          Test Configuration
        </h3>
        <p className="text-xs text-muted-foreground mb-4">
          Re-run this guardrail on recent failing logs to validate your changes
        </p>

        <div className="flex items-center gap-3">
          <Button
            onClick={handleRerun}
            disabled={rerunStatus === "running"}
          >
            {rerunStatus !== "running" && <PlayCircle className="h-4 w-4" />}
            {rerunStatus === "running"
              ? "Running on 10 samples..."
              : "Re-run on failing logs"}
          </Button>

          {rerunStatus === "success" && (
            <span className="text-sm text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
              <CheckCircle className="h-4 w-4" /> 7/10 would now pass with new config
            </span>
          )}

          {rerunStatus === "error" && (
            <span className="text-sm text-destructive">Error running tests</span>
          )}
        </div>
      </div>
    </div>
  );
}
