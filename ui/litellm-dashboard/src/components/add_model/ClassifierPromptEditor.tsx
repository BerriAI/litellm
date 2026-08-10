import React, { useCallback, useState } from "react";
import { TriangleAlert } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getAutoRouterClassifierDefaultPromptCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { hasCustomPrompt, initialDraftText, resolveCustomPrompt } from "./classifierPromptEditorState";
import { useTranslation } from "react-i18next";

interface ClassifierPromptEditorProps {
  systemPrompt: string | undefined;
  onChange: (systemPrompt: string | undefined) => void;
  contextWindowSize: number;
  tierLabels?: Record<string, string>;
}

const ClassifierPromptEditor: React.FC<ClassifierPromptEditorProps> = ({
  systemPrompt,
  onChange,
  contextWindowSize,
  tierLabels,
}) => {
  const { t } = useTranslation("gateway");
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
      const fetched = await getAutoRouterClassifierDefaultPromptCall(accessToken, contextWindowSize, tierLabels);
      setDefaultPrompt(fetched);
      setDraft(initialDraftText(systemPrompt, fetched));
    } catch {
      NotificationsManager.fromBackend(t("models.autoRouters.details.promptEditor.loadFailed"));
      setIsOpen(false);
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, contextWindowSize, systemPrompt, tierLabels]);

  const handleSave = () => {
    onChange(resolveCustomPrompt({ text: draft, defaultPrompt }));
    setIsOpen(false);
  };

  return (
    <div>
      <div className="flex items-center gap-2">
        <Button type="button" size="sm" variant="outline" onClick={openEditor} disabled={!accessToken}>
          {isOverridden
            ? t("models.autoRouters.details.promptEditor.edit")
            : t("models.autoRouters.details.promptEditor.change")}
        </Button>
        {isOverridden && (
          <Button type="button" size="sm" variant="link" onClick={() => onChange(undefined)}>
            {t("models.autoRouters.details.promptEditor.reset")}
          </Button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {isOverridden
          ? t("models.autoRouters.details.promptEditor.overriddenDescription")
          : t("models.autoRouters.details.promptEditor.defaultDescription")}
      </p>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("models.autoRouters.details.promptEditor.title")}</DialogTitle>
          </DialogHeader>

          <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            <p className="flex items-center gap-2 font-medium">
              <TriangleAlert className="size-4" aria-hidden />
              {t("models.autoRouters.details.promptEditor.warning")}
            </p>
            <p className="mt-2">{t("models.autoRouters.details.promptEditor.warningInjection")}</p>
            <p className="mt-2">{t("models.autoRouters.details.promptEditor.warningTiers")}</p>
            <p className="mt-2">{t("models.autoRouters.details.promptEditor.warningFallback")}</p>
          </div>

          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={16}
            disabled={isLoading}
            aria-label={t("models.autoRouters.details.promptEditor.aria")}
            className="mt-3 font-mono text-xs"
          />
          <div className="mt-2 flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              {t("models.autoRouters.details.promptEditor.prefilled", { size: contextWindowSize })}
            </p>
            <Button
              type="button"
              size="sm"
              variant="link"
              onClick={() => setDraft(defaultPrompt)}
              disabled={isLoading || draft === defaultPrompt}
            >
              {t("models.autoRouters.details.promptEditor.restore")}
            </Button>
          </div>

          <DialogFooter className="mt-4">
            <Button type="button" variant="outline" onClick={() => setIsOpen(false)}>
              {t("models.autoRouters.details.promptEditor.cancel")}
            </Button>
            <Button type="button" onClick={handleSave} disabled={isLoading || !draft.trim()}>
              {t("models.autoRouters.details.promptEditor.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ClassifierPromptEditor;
