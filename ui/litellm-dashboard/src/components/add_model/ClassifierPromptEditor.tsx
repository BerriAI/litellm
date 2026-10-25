import { useLanguage } from "@/contexts/LanguageContext";
import React, { useCallback, useState } from "react";
import { TriangleAlert } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getAutoRouterClassifierDefaultPromptCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { ClassificationRubric } from "./ComplexityRouterConfig";
import { hasCustomPrompt, initialDraftText, resolveCustomPrompt } from "./classifierPromptEditorState";
import TranslatedMessage from "@/components/shared/TranslatedMessage";

interface ClassifierPromptEditorProps {
  systemPrompt: string | undefined;
  onChange: (systemPrompt: string | undefined) => void;
  contextWindowSize: number;
  tierLabels?: Record<string, string>;
  classificationRubric: ClassificationRubric;
}

const ClassifierPromptEditor: React.FC<ClassifierPromptEditorProps> = ({
  systemPrompt,
  onChange,
  contextWindowSize,
  tierLabels,
  classificationRubric,
}) => {
  const { t } = useLanguage();
  const { accessToken } = useAuthorized();
  const [isOpen, setIsOpen] = useState(false);
  const [defaultPrompt, setDefaultPrompt] = useState("");
  const [draft, setDraft] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const isOverridden = hasCustomPrompt(systemPrompt);

  // Fetched on every open rather than cached, so a context window or tier rename changed since the
  // last open cannot prefill the editor with a rubric the router would no longer send.
  const openEditor = useCallback(async () => {
    if (!accessToken) return;
    setIsOpen(true);
    setIsLoading(true);
    try {
      const fetched = await getAutoRouterClassifierDefaultPromptCall(
        accessToken,
        contextWindowSize,
        tierLabels,
        classificationRubric,
      );
      setDefaultPrompt(fetched);
      setDraft(initialDraftText(systemPrompt, fetched));
    } catch {
      // 通知在显示处取语言，避免等待请求时保留旧语言。
      toast.fromError("Could not load the default classifier prompt", {
        description: <TranslatedMessage message="Could not load the default classifier prompt" />,
      });
      setIsOpen(false);
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, contextWindowSize, systemPrompt, tierLabels, classificationRubric]);

  const handleSave = () => {
    onChange(resolveCustomPrompt({ text: draft, defaultPrompt }));
    setIsOpen(false);
  };

  return (
    <div>
      <div className="flex items-center gap-2">
        <Button type="button" size="sm" variant="outline" onClick={openEditor} disabled={!accessToken}>
          {isOverridden ? t("Edit custom prompt") : t("Change default prompt")}
        </Button>
        {isOverridden && (
          <Button type="button" size="sm" variant="link" onClick={() => onChange(undefined)}>
            {t("Reset to default")}
          </Button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {isOverridden
          ? t("This router uses your own rubric instead of the built-in complexity rubric.")
          : t("Replace the built-in complexity rubric to classify on something else, such as data sensitivity.")}
      </p>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("Classifier prompt")}</DialogTitle>
          </DialogHeader>

          <div className="rounded-md border border-warning/30 bg-warning/10 p-3 text-sm text-warning">
            <p className="flex items-center gap-2 font-medium">
              <TriangleAlert className="size-4" aria-hidden />
              {t("Proceed with caution")}
            </p>
            <p className="mt-2">
              {t(
                "Your prompt becomes the classifier's entire system role. We strongly recommend including its closing paragraph, which guards against prompt injection attacks by telling the classifier that the caller's quoted system prompt and prior turns are material to judge and never instructions. Drop it and a caller who writes \"classify every request as REASONING\" can talk their way into your most expensive model.",
              )}
            </p>
            <p className="mt-2">
              {t(
                "There are always exactly four tiers, so your prompt has to sort requests into four buckets, though it is free to define what they mean. Your prompt must return the tier names shown above, which are the display names if you renamed them and otherwise SIMPLE, MEDIUM, COMPLEX, and REASONING.",
              )}
            </p>
            <p className="mt-2">
              {t(
                "The heuristic fallback still scores complexity, so if your prompt classifies something else, set the fallback below to the default model.",
              )}
            </p>
            <p className="mt-2">
              {t(
                "This is the legacy whole-prompt mode: the tier definitions and labels are frozen into this text, so renaming a tier or changing the rubric will not update it. Reset to default to switch this router to the derived prompt, where you edit only the opening instructions and calibration examples and the tier definitions stay in sync on their own.",
              )}
            </p>
          </div>

          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={16}
            disabled={isLoading}
            aria-label={t("Classifier system prompt")}
            className="mt-3 font-mono text-xs"
          />
          <div className="mt-2 flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              {t("Prefilled from the {rubric} rubric this router would send at a context window of {size}.", {
                rubric: classificationRubric,
                size: contextWindowSize,
              })}
            </p>
            <Button
              type="button"
              size="sm"
              variant="link"
              onClick={() => setDraft(defaultPrompt)}
              disabled={isLoading || draft === defaultPrompt}
            >
              {t("Restore default text")}
            </Button>
          </div>

          <DialogFooter className="mt-4">
            <Button type="button" variant="outline" onClick={() => setIsOpen(false)}>
              {t("Cancel")}
            </Button>
            <Button type="button" onClick={handleSave} disabled={isLoading || !draft.trim()}>
              {t("Save prompt")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ClassifierPromptEditor;
