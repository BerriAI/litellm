import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { ArrowLeftIcon, SaveIcon, ClockIcon, LoaderCircleIcon } from "lucide-react";
import PromptCodeSnippets from "./PromptCodeSnippets";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const ENVIRONMENT_ITEMS = [
  { value: "development", label: "Development" },
  { value: "staging", label: "Staging" },
  { value: "production", label: "Production" },
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
  const { t } = useLanguage();
  return (
    <div className="bg-background border-b border-border px-6 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <Button variant="ghost" onClick={onBack} size="sm">
          <ArrowLeftIcon />
          {t("Back")}
        </Button>
        <Input
          aria-label={t("Prompt name")}
          value={promptName}
          onChange={(e) => onNameChange(e.target.value)}
          className="text-base font-medium border-none shadow-none"
          style={{ width: "200px" }}
        />
        {version && <Badge>{version}</Badge>}
        <Select
          items={ENVIRONMENT_ITEMS}
          value={environment}
          onValueChange={(value) => onEnvironmentChange(String(value))}
        >
          <SelectTrigger size="sm" className="w-[140px]" aria-label={t("Environment")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ENVIRONMENT_ITEMS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Badge variant="secondary">{t("Draft")}</Badge>
        <span className="text-xs text-muted-foreground">{t("Unsaved changes")}</span>
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
            {t("History")}
          </Button>
        )}
        <Button onClick={onSave} disabled={isSaving}>
          {isSaving ? <LoaderCircleIcon className="animate-spin" /> : <SaveIcon />}
          {editMode ? t("Update") : t("Save")}
        </Button>
      </div>
    </div>
  );
};

export default PromptEditorHeader;
