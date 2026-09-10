import React, { useEffect, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getAutoRouterAssembledPromptCall } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { TierRow, tierDefinitionsFromRows } from "./tier_rows";
import {
  CLASSIFICATION_RUBRIC_DESCRIPTIONS,
  ClassificationRubric,
  ComplexityTierLabels,
} from "./ComplexityRouterConfig";

export type OpeningPromptTierSource =
  | { kind: "custom"; tierRows: readonly TierRow[] }
  | {
      kind: "builtIn";
      tierLabels?: ComplexityTierLabels;
      classificationRubric: ClassificationRubric;
      rubricRestriction?: string;
    };

/**
 * Everything the dialog can change, emitted together. The rubric rides the same payload as the two
 * text sections because the parent rebuilds its whole config value from one spread: two callbacks
 * fired in one tick would each start from the same stale value, so the second would drop the first.
 */
export interface OpeningPromptSelection {
  classificationPrompt: string | undefined;
  classificationExamples: string | undefined;
  classificationRubric?: ClassificationRubric;
}

interface OpeningPromptEditorProps {
  classificationPrompt: string | undefined;
  classificationExamples: string | undefined;
  onChange: (value: OpeningPromptSelection) => void;
  tierSource: OpeningPromptTierSource;
  contextWindowSize: number;
}

const CUSTOM_PLACEHOLDER = `Classify the request into exactly one tier for a payments engineering team.

Weigh what the request actually asks for, not how it is worded.`;

const BUILT_IN_PLACEHOLDER = `Classify the complexity of a user request into exactly one tier.

Judge the intellectual difficulty of answering correctly, not how short, long, or technical-sounding the request is.`;

const COPY = {
  custom: {
    overridden:
      "This router opens with your own instructions and calibration examples. Your tier definitions and the injection guard are still appended below them.",
    default:
      "Write the opening instructions and your own calibration examples. Your tier definitions and the injection guard are always appended below them.",
    explainer:
      "Your text is the opening of the classifier prompt, so it is where calibration examples of your own belong. The router appends your tier definitions and its injection guard underneath, and neither can be edited or removed from here. Edit the definitions themselves with Edit tiers above.",
    placeholder: CUSTOM_PLACEHOLDER,
  },
  builtIn: {
    overridden:
      "This router opens with your own instructions and calibration examples in place of the base rubric's. Its tier criteria and the injection guard are still appended below them.",
    default:
      "The base rubric supplies the opening instructions and calibration examples. Customize them to write your own; the tier criteria and the injection guard are always appended below them.",
    explainer:
      "The base rubric decides the tier criteria and, until you write your own, the opening instructions and calibration examples. Your text replaces that opening and those examples. The router appends the four tier criteria and its injection guard underneath, and neither can be edited or removed from here. Rename the tiers with the display names above.",
    placeholder: BUILT_IN_PLACEHOLDER,
  },
} as const;

const OpeningPromptEditor: React.FC<OpeningPromptEditorProps> = ({
  classificationPrompt,
  classificationExamples,
  onChange,
  tierSource,
  contextWindowSize,
}) => {
  const { accessToken } = useAuthorized();
  const [isOpen, setIsOpen] = useState(false);
  const [instructionDraft, setInstructionDraft] = useState("");
  const [exampleDraft, setExampleDraft] = useState("");
  const [rubricDraft, setRubricDraft] = useState<ClassificationRubric | undefined>(undefined);
  const [preview, setPreview] = useState<
    { status: "loading" } | { status: "error" } | { status: "ready"; text: string }
  >({ status: "loading" });
  const isOverridden = Boolean(classificationPrompt?.trim() || classificationExamples?.trim());
  const copy = COPY[tierSource.kind];

  // Depended on individually rather than through tierSource, whose object identity a parent render
  // rebuilds every time: the effect writes state, so an identity dep would refetch on its own write.
  const tierRows = tierSource.kind === "custom" ? tierSource.tierRows : undefined;
  const tierLabels = tierSource.kind === "builtIn" ? tierSource.tierLabels : undefined;
  const savedRubric = tierSource.kind === "builtIn" ? tierSource.classificationRubric : undefined;
  // The dialog previews the rubric being considered, so the picker edits a draft the same way the two
  // text sections do. Writing straight through would survive Cancel and change the live classifier.
  const classificationRubric = isOpen ? rubricDraft ?? savedRubric : savedRubric;
  const rubricSummary = savedRubric === undefined ? null : CLASSIFICATION_RUBRIC_DESCRIPTIONS[savedRubric];
  // The trigger names what is saved; the dialog describes what is being previewed, so the two read
  // from different rubrics while a pick is still a draft.
  const draftRubricSummary =
    classificationRubric === undefined ? null : CLASSIFICATION_RUBRIC_DESCRIPTIONS[classificationRubric];

  useEffect(() => {
    if (!isOpen || !accessToken) return;
    let stale = false;
    const timer = setTimeout(async () => {
      try {
        const text = await getAutoRouterAssembledPromptCall(
          accessToken,
          contextWindowSize,
          tierRows ? { tierDefinitions: tierDefinitionsFromRows(tierRows) } : { tierLabels, classificationRubric },
          { classificationPrompt: instructionDraft, classificationExamples: exampleDraft },
        );
        if (!stale) setPreview({ status: "ready", text });
      } catch {
        if (!stale) setPreview({ status: "error" });
      }
    }, 300);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  }, [
    isOpen,
    accessToken,
    contextWindowSize,
    tierRows,
    tierLabels,
    classificationRubric,
    instructionDraft,
    exampleDraft,
  ]);

  const openEditor = () => {
    setInstructionDraft(classificationPrompt ?? "");
    setExampleDraft(classificationExamples ?? "");
    setRubricDraft(savedRubric);
    setPreview({ status: "loading" });
    setIsOpen(true);
  };

  const handleSave = () => {
    onChange({
      ...(savedRubric !== undefined && { classificationRubric: rubricDraft ?? savedRubric }),
      classificationPrompt: instructionDraft.trim() || undefined,
      classificationExamples: exampleDraft.trim() || undefined,
    });
    setIsOpen(false);
  };

  return (
    <div>
      {rubricSummary && (
        <p className="mb-1 text-xs text-muted-foreground">
          {isOverridden ? `Custom opening on the ${rubricSummary.label} rubric` : `${rubricSummary.label} rubric`}
        </p>
      )}
      <div className="flex items-center gap-2">
        <Button type="button" size="sm" variant="outline" onClick={openEditor}>
          {isOverridden ? "Edit custom prompt" : "Customize prompt"}
        </Button>
        {isOverridden && (
          <Button
            type="button"
            size="sm"
            variant="link"
            onClick={() =>
              onChange({
                ...(savedRubric !== undefined && { classificationRubric: savedRubric }),
                classificationPrompt: undefined,
                classificationExamples: undefined,
              })
            }
          >
            Reset to default
          </Button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{isOverridden ? copy.overridden : copy.default}</p>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-4xl">
          <DialogHeader>
            <DialogTitle>Classifier prompt</DialogTitle>
          </DialogHeader>

          {tierSource.kind === "builtIn" && (
            <div>
              <label className="text-sm font-medium" htmlFor="base-classification-rubric">
                Base rubric
              </label>
              <Select
                items={Object.entries(CLASSIFICATION_RUBRIC_DESCRIPTIONS).map(([rubric, description]) => ({
                  value: rubric,
                  label: description.label,
                }))}
                value={classificationRubric ?? tierSource.classificationRubric}
                onValueChange={(rubric: ClassificationRubric | null) => rubric && setRubricDraft(rubric)}
                disabled={Boolean(tierSource.rubricRestriction)}
              >
                <SelectTrigger id="base-classification-rubric" aria-label="Base rubric" className="mt-1 w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent
                  align="start"
                  data-testid="base-rubric-menu"
                  style={{ width: "24rem", maxWidth: "calc(100vw - 2rem)" }}
                >
                  {Object.entries(CLASSIFICATION_RUBRIC_DESCRIPTIONS).map(([rubric, description]) => (
                    <SelectItem key={rubric} value={rubric}>
                      {description.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="mt-1 text-xs text-muted-foreground">
                {tierSource.rubricRestriction ?? draftRubricSummary?.description}
              </p>
            </div>
          )}

          <p className="text-sm text-muted-foreground">{copy.explainer}</p>

          <div className="mt-3 space-y-4">
            <div>
              <label className="text-sm font-medium" htmlFor="classification-instructions">
                Classification instructions
              </label>
              <p className="mt-1 text-xs text-muted-foreground">
                Explain what the classifier should judge. Tier definitions are managed separately below.
              </p>
              <Textarea
                id="classification-instructions"
                value={instructionDraft}
                onChange={(e) => setInstructionDraft(e.target.value)}
                rows={5}
                placeholder={copy.placeholder}
                aria-label="Classification instructions"
                className="mt-2 font-mono text-xs"
              />
            </div>

            <div>
              <label className="text-sm font-medium" htmlFor="calibration-examples">
                Calibration examples
              </label>
              <p className="mt-1 text-xs text-muted-foreground">
                Show representative requests and the tier they should receive. The router adds these after its tier
                definitions.
              </p>
              <Textarea
                id="calibration-examples"
                value={exampleDraft}
                onChange={(e) => setExampleDraft(e.target.value)}
                rows={6}
                placeholder={'- "what is the capital of France?" -> SIMPLE'}
                aria-label="Calibration examples"
                className="mt-2 font-mono text-xs"
              />
            </div>
          </div>

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

export default OpeningPromptEditor;
