import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import React from "react";
import { ModelGroup } from "../playground/llm_calls/fetch_models";

interface ComplexityTiers {
  SIMPLE: string;
  MEDIUM: string;
  COMPLEX: string;
  REASONING: string;
}

interface ComplexityRouterConfigProps {
  modelInfo: ModelGroup[];
  value: ComplexityTiers;
  onChange: (tiers: ComplexityTiers) => void;
}

const TIER_DESCRIPTIONS: Record<
  keyof ComplexityTiers,
  { label: string; description: string; examples: string }
> = {
  SIMPLE: {
    label: "Simple",
    description: "Basic questions, greetings, simple factual queries",
    examples: '"Hello!", "What is Python?", "Thanks!"',
  },
  MEDIUM: {
    label: "Medium",
    description: "Standard queries requiring some reasoning or explanation",
    examples: '"Explain how REST APIs work", "Debug this error"',
  },
  COMPLEX: {
    label: "Complex",
    description: "Technical, multi-part requests requiring deep knowledge",
    examples:
      '"Design a microservices architecture", "Implement a rate limiter"',
  },
  REASONING: {
    label: "Reasoning",
    description: "Chain-of-thought, analysis, explicit reasoning requests",
    examples: '"Think step by step...", "Analyze the pros and cons..."',
  },
};

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({
  modelInfo,
  value,
  onChange,
}) => {
  const modelOptions = modelInfo.map((model) => ({
    value: model.model_group,
    label: model.model_group,
  }));

  const handleTierChange = (tier: keyof ComplexityTiers, model: string) => {
    onChange({
      ...value,
      [tier]: model,
    });
  };

  return (
    <div className="w-full max-w-none">
      <div className="flex items-center gap-2 mb-4">
        <h4 className="text-lg font-semibold m-0">
          Complexity Tier Configuration
        </h4>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-3.5 w-3.5 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              Map each complexity tier to a model. Simple queries use
              cheaper/faster models, complex queries use more capable models.
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      <p className="text-sm text-muted-foreground block mb-6">
        The complexity router automatically classifies requests by complexity
        using rule-based scoring (no API calls, &lt;1ms latency). Configure
        which model handles each tier.
      </p>

      <Card className="p-4">
        {(Object.keys(TIER_DESCRIPTIONS) as Array<keyof ComplexityTiers>).map(
          (tier, index) => {
            const tierInfo = TIER_DESCRIPTIONS[tier];
            return (
              <div key={tier}>
                {index > 0 && <hr className="my-4 border-border" />}
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-bold text-base">
                      {tierInfo.label} Tier
                    </span>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          {tierInfo.description}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <p className="text-xs text-muted-foreground block mb-2">
                    Examples: {tierInfo.examples}
                  </p>
                  <Select
                    value={value[tier]}
                    onValueChange={(model) => handleTierChange(tier, model)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue
                        placeholder={`Select model for ${tierInfo.label.toLowerCase()} queries`}
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {modelOptions.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value}>
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            );
          },
        )}
      </Card>

      <hr className="my-4 border-border" />

      <Card className="bg-muted p-4">
        <p className="font-bold block mb-2">How Classification Works</p>
        <p className="text-[13px] text-muted-foreground">
          The router scores each request across 7 dimensions: token count, code
          presence, reasoning markers, technical terms, simple indicators,
          multi-step patterns, and question complexity. The weighted score
          determines the tier:
        </p>
        <ul className="mt-2 mb-0 pl-5 text-[13px] text-muted-foreground">
          <li>
            <strong>SIMPLE</strong>: Score &lt; 0.15
          </li>
          <li>
            <strong>MEDIUM</strong>: Score 0.15 - 0.35
          </li>
          <li>
            <strong>COMPLEX</strong>: Score 0.35 - 0.60
          </li>
          <li>
            <strong>REASONING</strong>: Score &gt; 0.60 (or 2+ reasoning
            markers)
          </li>
        </ul>
      </Card>
    </div>
  );
};

export default ComplexityRouterConfig;
