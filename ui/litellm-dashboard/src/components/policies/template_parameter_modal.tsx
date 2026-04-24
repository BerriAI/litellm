import React, { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  RadioGroup,
  RadioGroupItem,
} from "@/components/ui/radio-group";
import { LoaderCircle, X } from "lucide-react";
import {
  modelHubCall,
  enrichPolicyTemplateStream,
} from "../networking";

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
    enrichmentOptions?: { model?: string; competitors?: string[] }
  ) => void;
  onCancel: () => void;
  isLoading?: boolean;
  accessToken: string;
}

function TagChips({
  value,
  onChange,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  placeholder: string;
}) {
  const [draft, setDraft] = useState("");
  const commit = (raw: string) => {
    const parts = raw
      .split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    const next = [...value];
    for (const p of parts) {
      if (!next.includes(p)) next.push(p);
    }
    onChange(next);
    setDraft("");
  };
  return (
    <div className="space-y-2">
      <Input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            if (draft.trim()) commit(draft);
          } else if (e.key === "Backspace" && !draft && value.length > 0) {
            onChange(value.slice(0, -1));
          }
        }}
        onBlur={() => {
          if (draft.trim()) commit(draft);
        }}
      />
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((v) => (
            <Badge
              key={v}
              variant="secondary"
              className="flex items-center gap-1"
            >
              {v}
              <button
                type="button"
                onClick={() => onChange(value.filter((x) => x !== v))}
                className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                aria-label={`Remove ${v}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

const TemplateParameterModal: React.FC<TemplateParameterModalProps> = ({
  visible,
  template,
  onConfirm,
  onCancel,
  isLoading = false,
  accessToken,
}) => {
  const [parameterValues, setParameterValues] = useState<
    Record<string, string>
  >({});
  const [competitorMode, setCompetitorMode] = useState<"ai" | "manual">("ai");
  const [selectedModel, setSelectedModel] = useState<string | undefined>(
    undefined,
  );
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [competitorTags, setCompetitorTags] = useState<string[]>([]);
  const [variationsMap, setVariationsMap] = useState<Record<string, string[]>>(
    {},
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [refinementInput, setRefinementInput] = useState("");
  const [isRefining, setIsRefining] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const parameters: TemplateParameter[] = template?.parameters || [];
  const hasEnrichment = !!template?.llm_enrichment;
  const enrichmentParam = hasEnrichment
    ? template.llm_enrichment.parameter
    : null;

  const nonEnrichmentParams = hasEnrichment
    ? parameters.filter((p) => p.name !== enrichmentParam)
    : parameters;

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
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, template]);

  useEffect(() => {
    if (
      visible &&
      hasEnrichment &&
      competitorMode === "ai" &&
      availableModels.length === 0
    ) {
      loadModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, hasEnrichment, competitorMode]);

  const loadModels = async () => {
    if (!accessToken) return;
    setIsLoadingModels(true);
    try {
      const fetchedModels = await modelHubCall(accessToken);
      if (fetchedModels?.data?.length > 0) {
        const models = fetchedModels.data
          .map((item: any) => item.model_group as string)
          .sort();
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
    const brandName = (
      parameterValues[enrichmentParam || "brand_name"] || ""
    ).trim();
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
    if (
      !accessToken ||
      !selectedModel ||
      !template ||
      !refinementInput.trim()
    )
      return;

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
            if (prev.some((t) => t.toLowerCase() === name.toLowerCase()))
              return prev;
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

  const brandNameFilled = enrichmentParam
    ? (parameterValues[enrichmentParam] || "").trim().length > 0
    : true;

  const canContinue = hasEnrichment
    ? allNonEnrichmentFilled && brandNameFilled && competitorTags.length > 0
    : allNonEnrichmentFilled && brandNameFilled;

  const handleConfirm = () => {
    onConfirm(parameterValues, { competitors: competitorTags });
  };

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? onCancel() : undefined)}
    >
      <DialogContent className="sm:max-w-[700px] max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{template?.title}</DialogTitle>
          <p className="text-sm text-muted-foreground font-normal">
            Configure competitor blocking for your brand
          </p>
        </DialogHeader>

        <div className="py-2 space-y-4">
          {nonEnrichmentParams.map((param) => (
            <div key={param.name} className="space-y-1.5">
              <label className="block text-sm font-medium">
                {param.label}
                {param.required && (
                  <span className="text-destructive ml-1">*</span>
                )}
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
          ))}

          {hasEnrichment && (
            <>
              <div className="space-y-2">
                <label className="block text-sm font-medium">
                  Competitor Discovery
                </label>
                <RadioGroup
                  value={competitorMode}
                  onValueChange={(v) =>
                    setCompetitorMode(v as "ai" | "manual")
                  }
                  className="flex gap-3"
                >
                  <label className="flex-1 cursor-pointer flex items-center justify-center gap-2 border border-border rounded-md py-2 data-[state=checked]:border-primary has-[:checked]:bg-accent">
                    <RadioGroupItem value="ai" />
                    ✨ Use AI
                  </label>
                  <label className="flex-1 cursor-pointer flex items-center justify-center gap-2 border border-border rounded-md py-2 has-[:checked]:bg-accent">
                    <RadioGroupItem value="manual" />
                    Enter Manually
                  </label>
                </RadioGroup>
              </div>

              <div className="space-y-1.5">
                <label className="block text-sm font-medium">
                  Your Brand Name
                  <span className="text-destructive ml-1">*</span>
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
                  <div className="space-y-1.5">
                    <label className="block text-sm font-medium">
                      Select Model
                      <span className="text-destructive ml-1">*</span>
                    </label>
                    <Select
                      value={selectedModel ?? ""}
                      onValueChange={(v) => setSelectedModel(v || undefined)}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue
                          placeholder={
                            isLoadingModels
                              ? "Loading models..."
                              : "Select a model to generate names"
                          }
                        />
                      </SelectTrigger>
                      <SelectContent>
                        {availableModels.length === 0 ? (
                          <div className="py-2 px-3 text-sm text-muted-foreground">
                            No models available
                          </div>
                        ) : (
                          availableModels.map((m) => (
                            <SelectItem key={m} value={m}>
                              {m}
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  <Button
                    onClick={handleGenerateNames}
                    disabled={
                      !selectedModel || !brandNameFilled || isGenerating
                    }
                    className="w-full"
                  >
                    {isGenerating ? (
                      <>
                        <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                        ✨ Generating names...
                      </>
                    ) : (
                      "✨ Generate Competitor Names"
                    )}
                  </Button>
                </>
              )}

              <div className="space-y-1.5">
                <label className="block text-sm font-medium">
                  Competitor Names
                  {competitorTags.length > 0 && (
                    <span className="text-muted-foreground font-normal ml-2">
                      ({competitorTags.length})
                    </span>
                  )}
                </label>
                <TagChips
                  value={competitorTags}
                  onChange={setCompetitorTags}
                  placeholder="Type a name and press Enter to add"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Type a name and press Enter to add. Click ✕ to remove.
                </p>
                {statusMessage && (
                  <div
                    // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
                    className="flex items-center gap-2 mt-2 p-2 bg-blue-50 rounded border border-blue-100"
                  >
                    {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                    <LoaderCircle className="w-3 h-3 animate-spin text-blue-700" />
                    {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                    <span className="text-xs text-blue-700">
                      {statusMessage}
                    </span>
                  </div>
                )}
                {Object.keys(variationsMap).length > 0 && !statusMessage && (
                  // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
                  <p className="text-xs text-green-600 mt-1">
                    ✓ {Object.values(variationsMap).flat().length} alternate
                    spellings & variations auto-generated for guardrail matching
                  </p>
                )}
              </div>

              {competitorMode === "ai" &&
                hasGenerated &&
                competitorTags.length > 0 && (
                  <div className="space-y-1.5">
                    <label className="block text-sm font-medium">
                      Refine List
                    </label>
                    <div className="flex gap-2">
                      <Input
                        placeholder="e.g. add 10 more from Asia, increase to 50 total..."
                        value={refinementInput}
                        onChange={(e) => setRefinementInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            refinementInput.trim() &&
                            !isRefining
                          ) {
                            e.preventDefault();
                            handleRefine();
                          }
                        }}
                        disabled={isRefining}
                      />
                      <Button
                        onClick={handleRefine}
                        disabled={!refinementInput.trim() || isRefining}
                        size="sm"
                      >
                        {isRefining ? (
                          <LoaderCircle className="h-3 w-3 animate-spin" />
                        ) : (
                          "Send"
                        )}
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Give instructions to add, remove, or change competitors.
                      Press Enter to send.
                    </p>
                  </div>
                )}
            </>
          )}

          {!hasEnrichment &&
            parameters.map((param) => (
              <div key={param.name} className="space-y-1.5">
                <label className="block text-sm font-medium">
                  {param.label}
                  {param.required && (
                    <span className="text-destructive ml-1">*</span>
                  )}
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
            ))}
        </div>

        <DialogFooter>
          <Button variant="secondary" onClick={onCancel} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={!canContinue || isLoading}
          >
            {isLoading ? (
              <>
                <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                Creating guardrails...
              </>
            ) : (
              "Continue"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default TemplateParameterModal;
