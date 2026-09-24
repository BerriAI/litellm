import React from "react";
import { toast } from "@/lib/toast";
import { Code, Info, TriangleAlert } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface CodeInterpreterToolProps {
  accessToken: string;
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  selectedContainerId: string | null;
  onContainerChange: (containerId: string | null) => void;
  selectedModel: string;
  disabled?: boolean;
}

const GITHUB_FEATURE_REQUEST_URL = "https://github.com/BerriAI/litellm/issues/new?template=feature_request.yml";

const isOpenAIModel = (model: string): boolean => {
  if (!model) return false;
  const lowerModel = model.toLowerCase();
  return (
    lowerModel.startsWith("openai/") ||
    lowerModel.startsWith("gpt-") ||
    lowerModel.startsWith("o1") ||
    lowerModel.startsWith("o3") ||
    lowerModel.includes("openai")
  );
};

const CodeInterpreterTool: React.FC<CodeInterpreterToolProps> = ({
  enabled,
  onEnabledChange,
  selectedModel,
  disabled = false,
}) => {
  const isOpenAI = isOpenAIModel(selectedModel);
  const isDisabled = disabled || !isOpenAI;

  const handleToggle = (checked: boolean) => {
    if (checked && !isOpenAI) {
      toast.warning("Code Interpreter is only available for OpenAI models");
      return;
    }
    onEnabledChange(checked);
  };

  return (
    <div className="border border-border rounded-lg p-3 bg-linear-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Code className="size-4 text-info" />
          <span className="font-medium text-foreground">Code Interpreter</span>
          <Tooltip>
            <TooltipTrigger aria-label="About Code Interpreter">
              <Info className="size-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              Run Python code to generate files, charts, and analyze data. Container is created automatically.
            </TooltipContent>
          </Tooltip>
        </div>
        <Switch
          checked={enabled && isOpenAI}
          onCheckedChange={handleToggle}
          disabled={isDisabled}
          size="sm"
          aria-label="Enable Code Interpreter"
        />
      </div>

      {!isOpenAI && (
        <div className="mt-2 pt-2 border-t border-border">
          <div className="flex items-start gap-2">
            <TriangleAlert className="mt-0.5 size-4 shrink-0 text-warning" />
            <div className="text-xs text-muted-foreground">
              <span>Code Interpreter is currently only supported for OpenAI models. </span>
              <a
                href={GITHUB_FEATURE_REQUEST_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-info hover:text-info/80 underline"
              >
                Request support for other providers
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeInterpreterTool;
