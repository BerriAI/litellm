import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Copy, Info } from "lucide-react";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import GuardrailTestResults from "./GuardrailTestResults";

interface GuardrailTestPanelProps {
  guardrailNames: string[];
  onSubmit: (text: string, metadata?: Record<string, unknown> | null) => void;
  isLoading: boolean;
  results: Array<{ guardrailName: string; response_text: string; latency: number }> | null;
  errors: Array<{ guardrailName: string; error: Error; latency: number }> | null;
  onClose: () => void;
}

export function GuardrailTestPanel({
  guardrailNames,
  onSubmit,
  isLoading,
  results,
  errors,
  onClose,
}: GuardrailTestPanelProps) {
  const { t } = useTranslation();
  const [inputText, setInputText] = useState("");
  const [metadataText, setMetadataText] = useState("");
  const [metadataError, setMetadataError] = useState<string | null>(null);

  const parseMetadata = (raw: string): { metadata: Record<string, unknown> | null; error: string | null } => {
    if (!raw.trim()) {
      return { metadata: null, error: null };
    }
    try {
      const parsed = JSON.parse(raw);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        return { metadata: null, error: t("guardrails.testPanel.errors.metadataNotObject") };
      }
      return { metadata: parsed, error: null };
    } catch {
      return { metadata: null, error: t("guardrails.testPanel.errors.invalidJson") };
    }
  };

  const handleSubmit = () => {
    if (!inputText.trim()) {
      toast.fromError(t("guardrails.testPanel.toast.emptyText"));
      return;
    }

    const { metadata, error } = parseMetadata(metadataText);
    if (error) {
      setMetadataError(error);
      toast.fromError(t("guardrails.testPanel.toast.metadataError", { error }));
      return;
    }
    setMetadataError(null);

    onSubmit(inputText, metadata);
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
      toast.success(t("guardrails.testPanel.toast.copied"));
    } else {
      toast.fromError(t("guardrails.testPanel.toast.copyFailed"));
    }
  };

  return (
    <div className="space-y-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border pb-3">
        <div className="flex items-center space-x-3">
          <div className="flex-1 min-w-0">
            <div className="mb-1 flex items-center space-x-2">
              <h2 className="text-lg font-semibold">{t("guardrails.testPanel.title")}</h2>
              <div className="flex flex-wrap gap-2">
                {guardrailNames.map((name) => (
                  <div
                    key={name}
                    className="inline-flex items-center space-x-1 rounded-md border border-info/20 bg-info/10 px-3 py-1"
                  >
                    <span className="font-mono text-sm font-medium text-info">{name}</span>
                  </div>
                ))}
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              <Trans
                i18nKey="guardrails.testPanel.subtitle"
                count={guardrailNames.length}
                components={{ noun: <></> }}
              />
            </p>
          </div>
        </div>
      </div>

      {/* Input Section */}
      <div className="flex-1 space-y-4 overflow-auto px-1">
        <div className="space-y-3">
          <div>
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">{t("guardrails.testPanel.inputLabel")}</label>
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <span className="cursor-help text-muted-foreground">
                        <Info className="size-3.5" />
                      </span>
                    }
                  />
                  <TooltipContent>{t("guardrails.testPanel.inputTooltip")}</TooltipContent>
                </Tooltip>
              </div>
              {inputText && (
                <Button size="sm" variant="secondary" onClick={handleCopyInput}>
                  <Copy />
                  {t("guardrails.testPanel.copyInput")}
                </Button>
              )}
            </div>
            <Textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t("guardrails.testPanel.inputPlaceholder")}
              rows={8}
              className="font-mono text-sm field-sizing-fixed"
            />
            <div className="mt-1 flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                <Trans i18nKey="guardrails.testPanel.keyboardHint" components={{ kbd: <Kbd /> }} />
              </span>
              <span className="text-xs text-muted-foreground">
                <Trans
                  i18nKey="guardrails.testPanel.characters"
                  components={{ length: <>{String(inputText.length)}</> }}
                />
              </span>
            </div>
          </div>

          <div>
            <div className="mb-2 flex items-center gap-2">
              <label className="text-sm font-medium">{t("guardrails.testPanel.metadataLabel")}</label>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <span className="cursor-help text-muted-foreground">
                      <Info className="size-3.5" />
                    </span>
                  }
                />
                <TooltipContent>{t("guardrails.testPanel.metadataTooltip")}</TooltipContent>
              </Tooltip>
            </div>
            <Textarea
              value={metadataText}
              onChange={(e) => {
                setMetadataText(e.target.value);
                if (metadataError) {
                  setMetadataError(parseMetadata(e.target.value).error);
                }
              }}
              placeholder='{"forbidden_topics": ["tax", "finance"]}'
              rows={3}
              className="font-mono text-sm field-sizing-fixed"
              aria-invalid={metadataError ? true : undefined}
            />
            {metadataError && <span className="text-xs text-destructive">{metadataError}</span>}
          </div>

          <div className="pt-2">
            <Button
              onClick={handleSubmit}
              disabled={!inputText.trim() || isLoading}
              aria-busy={isLoading}
              className="w-full"
            >
              {isLoading && <UiLoadingSpinner className="size-4" />}
              {isLoading
                ? t("guardrails.testPanel.submit.running", { count: guardrailNames.length })
                : t("guardrails.testPanel.submit.idle", { count: guardrailNames.length })}
            </Button>
          </div>
        </div>

        {/* Results Section */}
        <GuardrailTestResults results={results} errors={errors} />
      </div>
    </div>
  );
}

export default GuardrailTestPanel;
