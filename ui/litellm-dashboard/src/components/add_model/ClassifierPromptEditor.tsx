import React, { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { TriangleAlert } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getAutoRouterClassifierDefaultPromptCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { ClassificationRubric } from "./ComplexityRouterConfig";
import { hasCustomPrompt, initialDraftText, resolveCustomPrompt } from "./classifierPromptEditorState";

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
  const { t } = useTranslation();
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
      toast.fromError(i18n.t("addModel.classifierPromptEditor.loadError"));
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
          {isOverridden
            ? t("addModel.classifierPromptEditor.editCustomPrompt")
            : t("addModel.classifierPromptEditor.changeDefaultPrompt")}
        </Button>
        {isOverridden && (
          <Button type="button" size="sm" variant="link" onClick={() => onChange(undefined)}>
            {t("addModel.classifierPromptEditor.resetToDefault")}
          </Button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {isOverridden
          ? t("addModel.classifierPromptEditor.overriddenHelp")
          : t("addModel.classifierPromptEditor.defaultHelp")}
      </p>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("addModel.classifierPromptEditor.dialogTitle")}</DialogTitle>
          </DialogHeader>

          <div className="rounded-md border border-warning/30 bg-warning/10 p-3 text-sm text-warning">
            <p className="flex items-center gap-2 font-medium">
              <TriangleAlert className="size-4" aria-hidden />
              {t("addModel.classifierPromptEditor.caution.title")}
            </p>
            <p className="mt-2">{t("addModel.classifierPromptEditor.caution.injectionGuard")}</p>
            <p className="mt-2">{t("addModel.classifierPromptEditor.caution.fourTiers")}</p>
            <p className="mt-2">{t("addModel.classifierPromptEditor.caution.heuristicFallback")}</p>
            <p className="mt-2">{t("addModel.classifierPromptEditor.caution.legacyMode")}</p>
          </div>

          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={16}
            disabled={isLoading}
            aria-label={t("addModel.classifierPromptEditor.systemPromptAriaLabel")}
            className="mt-3 font-mono text-xs"
          />
          <div className="mt-2 flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              {t("addModel.classifierPromptEditor.prefilledFrom", { rubric: classificationRubric, contextWindowSize })}
            </p>
            <Button
              type="button"
              size="sm"
              variant="link"
              onClick={() => setDraft(defaultPrompt)}
              disabled={isLoading || draft === defaultPrompt}
            >
              {t("addModel.classifierPromptEditor.restoreDefaultText")}
            </Button>
          </div>

          <DialogFooter className="mt-4">
            <Button type="button" variant="outline" onClick={() => setIsOpen(false)}>
              {t("addModel.classifierPromptEditor.cancel")}
            </Button>
            <Button type="button" onClick={handleSave} disabled={isLoading || !draft.trim()}>
              {t("addModel.classifierPromptEditor.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ClassifierPromptEditor;
