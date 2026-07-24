import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { X } from "lucide-react";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { modelHubCall, enrichPolicyTemplateStream } from "@/components/networking";

interface TemplateParameter {
  name: string;
  label: string;
  type: string;
  required: boolean;
  placeholder?: string;
}

interface TemplateParameterModalProps {
  visible: boolean;
  template: any;
  onConfirm: (
    parameters: Record<string, string>,
    enrichmentOptions?: { model?: string; competitors?: string[] },
  ) => void;
  onCancel: () => void;
  isLoading?: boolean;
  accessToken: string;
}

const TemplateParameterModal: React.FC<TemplateParameterModalProps> = ({
  visible,
  template,
  onConfirm,
  onCancel,
  isLoading = false,
  accessToken,
}) => {
  const [parameterValues, setParameterValues] = useState<Record<string, string>>({});
  const [competitorMode, setCompetitorMode] = useState<"ai" | "manual">("ai");
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [competitorTags, setCompetitorTags] = useState<string[]>([]);
  const [variationsMap, setVariationsMap] = useState<Record<string, string[]>>({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [refinementInput, setRefinementInput] = useState("");
  const [isRefining, setIsRefining] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [tagDraft, setTagDraft] = useState("");

  const parameters: TemplateParameter[] = template?.parameters || [];
  const hasEnrichment = !!template?.llm_enrichment;
  const enrichmentParam = hasEnrichment ? template.llm_enrichment.parameter : null;

  const nonEnrichmentParams = hasEnrichment ? parameters.filter((p) => p.name !== enrichmentParam) : parameters;

  useEffect(() => {
    if (visible && template) {
      const initial: Record<string, string> = {};
      parameters.forEach((p) => {
        initial[p.name] = "";
      });
      setParameterValues(initial);
      setCompetitorMode("ai");
      setSelectedModel(undefined);
      setCompetitorTags([]);
      setVariationsMap({});
      setIsGenerating(false);
      setRefinementInput("");
      setIsRefining(false);
      setHasGenerated(false);
      setStatusMessage("");
      setTagDraft("");
    }
  }, [visible, template]);

  useEffect(() => {
    if (visible && hasEnrichment && competitorMode === "ai" && availableModels.length === 0) {
      loadModels();
    }
  }, [visible, hasEnrichment, competitorMode]);

  const loadModels = async () => {
    if (!accessToken) return;
    setIsLoadingModels(true);
    try {
      const fetchedModels = await modelHubCall(accessToken);
      if (fetchedModels?.data?.length > 0) {
        const models = fetchedModels.data.map((item: any) => item.model_group as string).sort();
        setAvailableModels(models);
      }
    } catch (error) {
      console.error("Error fetching models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const handleGenerateNames = async () => {
    if (!accessToken || !selectedModel || !template) return;
    const brandName = (parameterValues[enrichmentParam || "brand_name"] || "").trim();
    if (!brandName) return;

    setIsGenerating(true);
    setCompetitorTags([]);
    setVariationsMap({});
    setStatusMessage("");
    try {
      await enrichPolicyTemplateStream(
        accessToken,
        template.id,
        parameterValues,
        selectedModel,
        (name) => {
          setCompetitorTags((prev) => [...prev, name]);
        },
        (result) => {
          setCompetitorTags(result.competitors);
          setVariationsMap(result.competitor_variations || {});
          setIsGenerating(false);
          setHasGenerated(true);
          setStatusMessage("");
        },
        (error) => {
          console.error("Streaming error:", error);
          setIsGenerating(false);
          setStatusMessage("");
        },
        undefined,
        (status) => setStatusMessage(status),
      );
    } catch (error) {
      console.error("Error generating competitor names:", error);
      setIsGenerating(false);
    }
  };

  const handleRefine = async () => {
    if (!accessToken || !selectedModel || !template || !refinementInput.trim()) return;

    setIsRefining(true);
    setStatusMessage("");
    try {
      await enrichPolicyTemplateStream(
        accessToken,
        template.id,
        parameterValues,
        selectedModel,
        (name) => {
          setCompetitorTags((prev) => {
            if (prev.some((t) => t.toLowerCase() === name.toLowerCase())) return prev;
            return [...prev, name];
          });
        },
        (result) => {
          setCompetitorTags(result.competitors);
          setVariationsMap(result.competitor_variations || {});
          setIsRefining(false);
          setRefinementInput("");
          setStatusMessage("");
        },
        (error) => {
          console.error("Refinement error:", error);
          setIsRefining(false);
          setStatusMessage("");
        },
        {
          instruction: refinementInput.trim(),
          existingCompetitors: competitorTags,
        },
        (status) => setStatusMessage(status),
      );
    } catch (error) {
      console.error("Error refining competitor names:", error);
      setIsRefining(false);
    }
  };

  const allNonEnrichmentFilled = nonEnrichmentParams
    .filter((p) => p.required)
    .every((p) => (parameterValues[p.name] || "").trim().length > 0);

  const brandNameFilled = enrichmentParam ? (parameterValues[enrichmentParam] || "").trim().length > 0 : true;

  const canContinue = hasEnrichment
    ? allNonEnrichmentFilled && brandNameFilled && competitorTags.length > 0
    : allNonEnrichmentFilled && brandNameFilled;

  const addCompetitorTags = (raw: string) => {
    const additions = raw
      .split(",")
      .map((name) => name.trim())
      .filter((name) => name.length > 0 && !competitorTags.some((t) => t.toLowerCase() === name.toLowerCase()));
    if (additions.length > 0) setCompetitorTags([...competitorTags, ...additions]);
    setTagDraft("");
  };

  const handleTagDraftKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addCompetitorTags(tagDraft);
      return;
    }
    if (e.key === "Backspace" && tagDraft === "" && competitorTags.length > 0) {
      setCompetitorTags(competitorTags.slice(0, -1));
    }
  };

  const handleConfirm = () => {
    onConfirm(parameterValues, { competitors: competitorTags });
  };

  const renderParameterField = (param: TemplateParameter) => (
    <div key={param.name}>
      <label className="mb-1 block text-sm font-medium">
        {param.label}
        {param.required && <span className="ml-1 text-destructive">*</span>}
      </label>
      <Input
        placeholder={param.placeholder || ""}
        value={parameterValues[param.name] || ""}
        onChange={(e) =>
          setParameterValues((prev) => ({
            ...prev,
            [param.name]: e.target.value,
          }))
        }
      />
    </div>
  );

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="sm:max-w-175">
        <DialogHeader>
          <DialogTitle className="text-lg">{template?.title}</DialogTitle>
          <DialogDescription>Configure competitor blocking for your brand</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {nonEnrichmentParams.map(renderParameterField)}

          {hasEnrichment && (
            <>
              <div>
                <label className="mb-2 block text-sm font-medium">Competitor Discovery</label>
                <RadioGroup
                  value={competitorMode}
                  onValueChange={(value) => setCompetitorMode(value as "ai" | "manual")}
                  className="grid-cols-2"
                >
                  <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-input px-3 py-2 text-sm">
                    <RadioGroupItem value="ai" />✨ Use AI
                  </label>
                  <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-input px-3 py-2 text-sm">
                    <RadioGroupItem value="manual" />
                    Enter Manually
                  </label>
                </RadioGroup>
              </div>

              {/* Brand Name */}
              <div>
                <label className="mb-1 block text-sm font-medium">
                  Your Brand Name
                  <span className="ml-1 text-destructive">*</span>
                </label>
                <Input
                  placeholder="e.g. Acme Airlines"
                  value={parameterValues[enrichmentParam || "brand_name"] || ""}
                  onChange={(e) =>
                    setParameterValues((prev) => ({
                      ...prev,
                      [enrichmentParam || "brand_name"]: e.target.value,
                    }))
                  }
                />
              </div>

              {competitorMode === "ai" && (
                <>
                  <div>
                    <label className="mb-1 block text-sm font-medium">
                      Select Model
                      <span className="ml-1 text-destructive">*</span>
                    </label>
                    <SearchSelect
                      options={availableModels.map((m) => ({ label: m, value: m }))}
                      value={selectedModel}
                      onValueChange={(value) => setSelectedModel(value || undefined)}
                      placeholder={isLoadingModels ? "Loading models..." : "Select a model to generate names"}
                      emptyText="No models found"
                      disabled={isLoadingModels}
                    />
                  </div>

                  <Button
                    onClick={handleGenerateNames}
                    disabled={!selectedModel || !brandNameFilled || isGenerating}
                    className="w-full"
                  >
                    {isGenerating ? "✨ Generating names..." : "✨ Generate Competitor Names"}
                  </Button>
                </>
              )}

              {/* Competitor Tags */}
              <div>
                <label className="mb-1 block text-sm font-medium">
                  Competitor Names
                  {competitorTags.length > 0 && (
                    <span className="ml-2 font-normal text-muted-foreground">({competitorTags.length})</span>
                  )}
                </label>
                <div className="flex flex-wrap items-center gap-1.5 rounded-md border border-input p-2">
                  {competitorTags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="gap-1">
                      {tag}
                      <button
                        type="button"
                        aria-label={`Remove ${tag}`}
                        onClick={() => setCompetitorTags(competitorTags.filter((t) => t !== tag))}
                      >
                        <X className="size-3" />
                      </button>
                    </Badge>
                  ))}
                  <input
                    className="min-w-40 flex-1 bg-transparent text-sm outline-none"
                    placeholder="Type a name and press Enter to add"
                    value={tagDraft}
                    onChange={(e) => setTagDraft(e.target.value)}
                    onKeyDown={handleTagDraftKeyDown}
                  />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Type a name and press Enter to add. Click ✕ to remove.
                </p>
                {statusMessage && (
                  <div className="mt-2 flex items-center gap-2 rounded-sm border border-border bg-muted p-2">
                    <UiLoadingSpinner className="size-3" />
                    <span className="text-xs text-muted-foreground">{statusMessage}</span>
                  </div>
                )}
                {Object.keys(variationsMap).length > 0 && !statusMessage && (
                  <p className="mt-1 text-xs text-green-600">
                    ✓ {Object.values(variationsMap).flat().length} alternate spellings &amp; variations auto-generated
                    for guardrail matching
                  </p>
                )}
              </div>

              {/* Refinement input — shown after initial generation in AI mode */}
              {competitorMode === "ai" && hasGenerated && competitorTags.length > 0 && (
                <div>
                  <label className="mb-1 block text-sm font-medium">Refine List</label>
                  <div className="flex gap-2">
                    <Input
                      placeholder="e.g. add 10 more from Asia, increase to 50 total..."
                      value={refinementInput}
                      onChange={(e) => setRefinementInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && refinementInput.trim() && !isRefining) {
                          handleRefine();
                        }
                      }}
                      disabled={isRefining}
                    />
                    <Button onClick={handleRefine} disabled={!refinementInput.trim() || isRefining} size="sm">
                      {isRefining ? "..." : "Send"}
                    </Button>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Give instructions to add, remove, or change competitors. Press Enter to send.
                  </p>
                </div>
              )}
            </>
          )}

          {!hasEnrichment && parameters.map(renderParameterField)}
        </div>

        <DialogFooter>
          <Button variant="secondary" onClick={onCancel} disabled={isLoading}>
            Cancel
          </Button>
          <Button onClick={handleConfirm} disabled={!canContinue || isLoading}>
            {isLoading ? "Creating guardrails..." : "Continue"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default TemplateParameterModal;
