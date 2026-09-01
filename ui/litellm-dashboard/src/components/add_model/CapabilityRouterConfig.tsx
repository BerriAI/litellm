import React from "react";
import { Plus, Trash2 } from "lucide-react";

import { type ModelGroup } from "@/components/llm_calls/fetch_models";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { type CapabilityRouterConfigValue } from "./capability_router_config";

interface Props {
  modelInfo: ModelGroup[];
  value: CapabilityRouterConfigValue;
  onChange: (value: CapabilityRouterConfigValue) => void;
}

const CapabilityRouterConfig: React.FC<Props> = ({ modelInfo, value, onChange }) => {
  const options = React.useMemo(
    () =>
      Array.from(new Set(modelInfo.filter((model) => model.mode !== "embedding").map((model) => model.model_group)))
        .filter((model) => !model.startsWith("auto_router/"))
        .map((model) => ({ value: model, label: model })),
    [modelInfo],
  );

  const updateCandidate = (index: number, patch: Partial<CapabilityRouterConfigValue["candidates"][number]>) => {
    const previousModel = value.candidates[index]?.model;
    const candidates = value.candidates.map((candidate, row) =>
      row === index ? { ...candidate, ...patch } : candidate,
    );
    const fallback_model =
      previousModel === value.fallback_model && patch.model !== undefined ? patch.model : value.fallback_model;
    onChange({ ...value, candidates, fallback_model });
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-base font-semibold">Candidate models</h3>
        <p className="text-sm text-muted-foreground">
          Describe the tasks each model can reliably complete. Price is calculated separately.
        </p>
      </div>
      {value.candidates.map((candidate, index) => (
        <div key={index} className="space-y-3 rounded-lg border border-border p-4">
          <div className="flex items-center justify-between">
            <Label>Candidate {index + 1}</Label>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              disabled={value.candidates.length <= 2}
              aria-label={`Remove candidate ${index + 1}`}
              onClick={() => {
                const candidates = value.candidates.filter((_, row) => row !== index);
                onChange({
                  ...value,
                  candidates,
                  fallback_model: candidates.some((item) => item.model === value.fallback_model)
                    ? value.fallback_model
                    : candidates[0]?.model ?? "",
                });
              }}
            >
              <Trash2 />
            </Button>
          </div>
          <SearchSelect
            options={options.filter(
              (option) =>
                option.value === candidate.model ||
                !value.candidates.some((item, row) => row !== index && item.model === option.value),
            )}
            value={candidate.model}
            onValueChange={(model) => updateCandidate(index, { model })}
            placeholder="Select a model group"
            emptyText="No models found"
          />
          <Textarea
            value={candidate.description}
            onChange={(event) => updateCandidate(index, { description: event.target.value })}
            placeholder="Example: Fast and reliable for extraction, summarization, and bounded code edits; weaker on long-horizon debugging."
            rows={3}
          />
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        onClick={() => onChange({ ...value, candidates: [...value.candidates, { model: "", description: "" }] })}
      >
        <Plus /> Add candidate
      </Button>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label>Classifier model</Label>
          <SearchSelect
            options={options}
            value={value.classifier.model}
            onValueChange={(model) => onChange({ ...value, classifier: { ...value.classifier, model } })}
            placeholder="Select classifier"
            emptyText="No models found"
          />
        </div>
        <div className="space-y-2">
          <Label>Fallback model</Label>
          <SearchSelect
            options={value.candidates
              .filter((candidate) => candidate.model)
              .map((candidate) => ({ value: candidate.model, label: candidate.model }))}
            value={value.fallback_model}
            onValueChange={(fallback_model) => onChange({ ...value, fallback_model })}
            placeholder="Select fallback"
            emptyText="Select candidates first"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="capability-probability-threshold">Probability threshold</Label>
          <Input
            id="capability-probability-threshold"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={value.probability_threshold}
            onChange={(event) => onChange({ ...value, probability_threshold: Number(event.target.value) })}
          />
          <p className="text-xs text-muted-foreground">
            Higher values favor reliability; lower values favor cost savings.
          </p>
        </div>
        <div className="space-y-2">
          <Label htmlFor="capability-cache-ttl">Decision cache TTL (seconds)</Label>
          <Input
            id="capability-cache-ttl"
            type="number"
            min={1}
            step={1}
            value={value.cache_ttl_seconds}
            onChange={(event) => onChange({ ...value, cache_ttl_seconds: Number(event.target.value) })}
          />
          <p className="text-xs text-muted-foreground">Reuses the decision for repeated calls in the same user turn.</p>
        </div>
      </div>
    </div>
  );
};

export default CapabilityRouterConfig;
