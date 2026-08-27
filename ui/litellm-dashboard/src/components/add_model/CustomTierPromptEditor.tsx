import React, { useCallback, useEffect, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getAutoRouterCustomTierPromptCall } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { TierRow, activeTierName } from "./tier_rows";

interface CustomTierPromptEditorProps {
  classificationPrompt: string | undefined;
  onChange: (classificationPrompt: string | undefined) => void;
  tierRows: readonly TierRow[];
  contextWindowSize: number;
}

const PLACEHOLDER = `Classify the request into exactly one tier for a payments engineering team.

Examples:
- "bump the copy on the checkout button" -> TRIAGE
- "why is our webhook signature check failing" -> SECURITY_REVIEW`;

const wireDefinitions = (tierRows: readonly TierRow[]): { name: string; description?: string }[] =>
  tierRows.map((row) => ({
    name: activeTierName(row),
    ...(row.definition.trim() && { description: row.definition.trim() }),
  }));

const CustomTierPromptEditor: React.FC<CustomTierPromptEditorProps> = ({
  classificationPrompt,
  onChange,
  tierRows,
  contextWindowSize,
}) => {
  const { accessToken } = useAuthorized();
  const [isOpen, setIsOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [preview, setPreview] = useState<
    { status: "loading" } | { status: "error" } | { status: "ready"; text: string }
  >({ status: "loading" });
  const isOverridden = Boolean(classificationPrompt?.trim());

  // Debounced so the preview follows the draft without a request per keystroke. Nothing is saved
  // from here, so a failed fetch leaves the preview empty rather than blocking the edit.
  const refreshPreview = useCallback(async () => {
    if (!accessToken) return;
    try {
      const text = await getAutoRouterCustomTierPromptCall(
        accessToken,
        contextWindowSize,
        wireDefinitions(tierRows),
        draft,
      );
      setPreview({ status: "ready", text });
    } catch {
      // Distinct from loading: a role that may not call the preview, or a prompt the write gate
      // would reject, otherwise leaves the panel claiming it is still fetching, forever.
      setPreview({ status: "error" });
    }
  }, [accessToken, contextWindowSize, tierRows, draft]);

  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(refreshPreview, 300);
    return () => clearTimeout(timer);
  }, [isOpen, refreshPreview]);

  const openEditor = () => {
    setDraft(classificationPrompt ?? "");
    setPreview({ status: "loading" });
    setIsOpen(true);
  };

  const handleSave = () => {
    onChange(draft.trim() || undefined);
    setIsOpen(false);
  };

  return (
    <div>
      <div className="flex items-center gap-2">
        <Button type="button" size="sm" variant="outline" onClick={openEditor}>
          Edit prompt
        </Button>
        {isOverridden && (
          <Button type="button" size="sm" variant="link" onClick={() => onChange(undefined)}>
            Reset to default
          </Button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {isOverridden
          ? "This router opens with your own instructions and calibration examples. Your tier definitions and the injection guard are still appended below them."
          : "Write the opening instructions and your own calibration examples. Your tier definitions and the injection guard are always appended below them."}
      </p>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-4xl">
          <DialogHeader>
            <DialogTitle>Classifier prompt</DialogTitle>
          </DialogHeader>

          <p className="text-sm text-muted-foreground">
            Your text is the opening of the classifier prompt, so it is where calibration examples of your own belong.
            The router appends your tier definitions and its injection guard underneath, and neither can be edited or
            removed from here. Edit the definitions themselves with Edit tiers above.
          </p>

          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={12}
            placeholder={PLACEHOLDER}
            aria-label="Classifier opening instructions"
            className="mt-3 font-mono text-xs"
          />

          <div className="mt-3">
            <p className="text-xs font-medium">What this router sends</p>
            {preview.status === "loading" && (
              <p className="mt-1 text-xs text-muted-foreground">Loading the assembled prompt…</p>
            )}
            {preview.status === "error" && (
              <p className="mt-1 text-xs text-muted-foreground">
                Could not load the assembled prompt. Your text is still saved as written.
              </p>
            )}
            {preview.status === "ready" && (
              <pre
                aria-label="Assembled classifier prompt"
                className="mt-1 overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs whitespace-pre-wrap text-muted-foreground"
              >
                {preview.text}
              </pre>
            )}
          </div>

          <DialogFooter className="mt-4">
            <Button type="button" variant="outline" onClick={() => setIsOpen(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={handleSave}>
              Save prompt
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CustomTierPromptEditor;
