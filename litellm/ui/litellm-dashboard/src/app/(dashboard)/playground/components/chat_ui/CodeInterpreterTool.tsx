import React from "react";
import { toast } from "@/lib/toast";
import { Code, Info, TriangleAlert } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
  const isOpenAI = isOpenAIModel(selectedModel);
  const isDisabled = disabled || !isOpenAI;

  const handleToggle = (checked: boolean) => {
    if (checked && !isOpenAI) {
      toast.warning(t("playground.chatUi.codeInterpreterTool.openAIOnlyToast"));
      return;
    }
    onEnabledChange(checked);
  };

  return (
    <div className="border border-border rounded-lg p-3 bg-linear-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Code className="size-4 text-info" />
          <span className="font-medium text-foreground">{t("playground.chatUi.codeInterpreterTool.title")}</span>
          <Tooltip>
            <TooltipTrigger aria-label={t("playground.chatUi.codeInterpreterTool.about")}>
              <Info className="size-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>{t("playground.chatUi.codeInterpreterTool.description")}</TooltipContent>
          </Tooltip>
        </div>
        <Switch
          checked={enabled && isOpenAI}
          onCheckedChange={handleToggle}
          disabled={isDisabled}
          size="sm"
          aria-label={t("playground.chatUi.codeInterpreterTool.enable")}
        />
      </div>

      {!isOpenAI && (
        <div className="mt-2 pt-2 border-t border-border">
          <div className="flex items-start gap-2">
            <TriangleAlert className="mt-0.5 size-4 shrink-0 text-warning" />
            <div className="text-xs text-muted-foreground">
              <span>{t("playground.chatUi.codeInterpreterTool.unsupported")}</span>
              <a
                href={GITHUB_FEATURE_REQUEST_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-info hover:text-info/80 underline"
              >
                {t("playground.chatUi.codeInterpreterTool.requestSupport")}
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeInterpreterTool;
