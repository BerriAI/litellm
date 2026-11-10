import React from "react";
import { ArrowLeftIcon, SaveIcon, ClockIcon, LoaderCircleIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import PromptCodeSnippets from "./PromptCodeSnippets";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const ENVIRONMENT_ITEMS = [
  { value: "development", labelKey: "prompts.editorHeader.environments.development" },
  { value: "staging", labelKey: "prompts.editorHeader.environments.staging" },
  { value: "production", labelKey: "prompts.editorHeader.environments.production" },
] as const;

interface PromptEditorHeaderProps {
  promptName: string;
  onNameChange: (name: string) => void;
  onBack: () => void;
  onSave: () => void;
  isSaving: boolean;
  editMode?: boolean;
  onShowHistory?: () => void;
  version?: string | null;
  promptModel?: string | null;
  promptVariables?: Record<string, string>;
  accessToken: string | null;
  proxySettings?: {
    PROXY_BASE_URL?: string;
    LITELLM_UI_API_DOC_BASE_URL?: string | null;
  };
  environment: string;
  onEnvironmentChange: (env: string) => void;
}

const PromptEditorHeader: React.FC<PromptEditorHeaderProps> = ({
  promptName,
  onNameChange,
  onBack,
  onSave,
  isSaving,
  editMode = false,
  onShowHistory,
  version,
  promptModel = "gpt-4o",
  promptVariables = {},
  accessToken,
  proxySettings,
  environment,
  onEnvironmentChange,
}) => {
  const { t } = useTranslation();
  return (
    <div className="bg-background border-b border-border px-6 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <Button variant="ghost" onClick={onBack} size="sm">
          <ArrowLeftIcon />
          {t("prompts.editorHeader.back")}
        </Button>
        <Input
          aria-label={t("prompts.editorHeader.nameLabel")}
          value={promptName}
          onChange={(e) => onNameChange(e.target.value)}
          className="text-base font-medium border-none shadow-none"
          style={{ width: "200px" }}
        />
        {version && <Badge>{version}</Badge>}
        <Select
          items={ENVIRONMENT_ITEMS.map((item) => ({ value: item.value, label: t(item.labelKey) }))}
          value={environment}
          onValueChange={(value) => onEnvironmentChange(String(value))}
        >
          <SelectTrigger size="sm" className="w-[140px]" aria-label={t("prompts.editorHeader.environmentLabel")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ENVIRONMENT_ITEMS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {t(item.labelKey)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Badge variant="secondary">{t("prompts.editorHeader.draft")}</Badge>
        <span className="text-xs text-muted-foreground">{t("prompts.editorHeader.unsaved")}</span>
      </div>
      <div className="flex items-center space-x-2">
        <PromptCodeSnippets
          promptId={promptName}
          model={promptModel ?? "YOUR_MODEL"}
          promptVariables={promptVariables}
          accessToken={accessToken}
          version={version?.replace("v", "") || "1"}
          environment={environment}
          proxySettings={proxySettings}
        />
        {editMode && onShowHistory && (
          <Button variant="outline" onClick={onShowHistory}>
            <ClockIcon />
            {t("prompts.editorHeader.history")}
          </Button>
        )}
        <Button onClick={onSave} disabled={isSaving}>
          {isSaving ? <LoaderCircleIcon className="animate-spin" /> : <SaveIcon />}
          {editMode ? t("prompts.editorHeader.update") : t("prompts.editorHeader.save")}
        </Button>
      </div>
    </div>
  );
};

export default PromptEditorHeader;
