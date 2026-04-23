import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Copy, Info } from "lucide-react";
import NotificationsManager from "../molecules/notifications_manager";
import GuardrailTestResults from "./GuardrailTestResults";

interface GuardrailTestPanelProps {
  guardrailNames: string[];
  onSubmit: (text: string) => void;
  isLoading: boolean;
  results:
    | Array<{ guardrailName: string; response_text: string; latency: number }>
    | null;
  errors:
    | Array<{ guardrailName: string; error: Error; latency: number }>
    | null;
  onClose: () => void;
}

export function GuardrailTestPanel({
  guardrailNames,
  onSubmit,
  isLoading,
  results,
  errors,
}: GuardrailTestPanelProps) {
  const [inputText, setInputText] = useState("");

  const handleSubmit = () => {
    if (!inputText.trim()) {
      NotificationsManager.fromBackend("Please enter text to test");
      return;
    }

    onSubmit(inputText);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);

        if (!successful) {
          throw new Error("execCommand failed");
        }
        return true;
      }
    } catch (error) {
      console.error("Copy failed:", error);
      return false;
    }
  };

  const handleCopyInput = async () => {
    const success = await copyToClipboard(inputText);
    if (success) {
      NotificationsManager.success("Input copied to clipboard");
    } else {
      NotificationsManager.fromBackend("Failed to copy input");
    }
  };

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <div className="flex items-center space-x-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h2 className="text-lg font-semibold text-foreground">
                Test Guardrails:
              </h2>
              <div className="flex flex-wrap gap-2">
                {guardrailNames.map((name) => (
                  <div
                    key={name}
                    className="inline-flex items-center space-x-1 bg-blue-50 dark:bg-blue-950/30 px-3 py-1 rounded-md border border-blue-200 dark:border-blue-900"
                  >
                    <span className="font-mono text-blue-700 dark:text-blue-300 font-medium text-sm">
                      {name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Test {guardrailNames.length > 1 ? "guardrails" : "guardrail"}{" "}
              and compare results
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto space-y-4">
        <div className="space-y-3">
          <div>
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-foreground">
                  Input Text
                </label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      Press Enter to submit. Use Shift+Enter for new line.
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              {inputText && (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={handleCopyInput}
                >
                  <Copy className="h-3.5 w-3.5" />
                  Copy Input
                </Button>
              )}
            </div>
            <Textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter text to test with guardrails..."
              rows={8}
              className="font-mono text-sm"
            />
            <div className="flex justify-between items-center mt-1">
              <span className="text-xs text-muted-foreground">
                Press{" "}
                <kbd className="px-1 py-0.5 bg-muted border border-border rounded text-xs">
                  Enter
                </kbd>{" "}
                to submit •{" "}
                <kbd className="px-1 py-0.5 bg-muted border border-border rounded text-xs">
                  Shift+Enter
                </kbd>{" "}
                for new line
              </span>
              <span className="text-xs text-muted-foreground">
                Characters: {inputText.length}
              </span>
            </div>
          </div>

          <div className="pt-2">
            <Button
              onClick={handleSubmit}
              disabled={isLoading || !inputText.trim()}
              className="w-full"
            >
              {isLoading
                ? `Testing ${guardrailNames.length} guardrail${guardrailNames.length > 1 ? "s" : ""}...`
                : `Test ${guardrailNames.length} guardrail${guardrailNames.length > 1 ? "s" : ""}`}
            </Button>
          </div>
        </div>

        <GuardrailTestResults results={results} errors={errors} />
      </div>
    </div>
  );
}

export default GuardrailTestPanel;
