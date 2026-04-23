import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowLeft, Clock, Save } from "lucide-react";
import PromptCodeSnippets from "./PromptCodeSnippets";

interface PromptEditorHeaderProps {
  promptName: string;
  onNameChange: (name: string) => void;
  onBack: () => void;
  onSave: () => void;
  isSaving: boolean;
  editMode?: boolean;
  onShowHistory?: () => void;
  version?: string | null;
  promptModel?: string;
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
  return (
    <div className="bg-background border-b border-border px-6 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="h-3.5 w-3.5" />
          Back
        </Button>
        <Input
          value={promptName}
          onChange={(e) => onNameChange(e.target.value)}
          className="text-base font-medium border-none shadow-none w-[200px] h-8"
        />
        {version && (
          <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 rounded font-medium">
            {version}
          </span>
        )}
        <Select value={environment} onValueChange={onEnvironmentChange}>
          <SelectTrigger className="w-[140px] h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="development">Development</SelectItem>
            <SelectItem value="staging">Staging</SelectItem>
            <SelectItem value="production">Production</SelectItem>
          </SelectContent>
        </Select>
        <span className="px-2 py-0.5 text-xs bg-muted text-muted-foreground rounded">
          Draft
        </span>
        <span className="text-xs text-muted-foreground">Unsaved changes</span>
      </div>
      <div className="flex items-center space-x-2">
        <PromptCodeSnippets
          promptId={promptName}
          model={promptModel}
          promptVariables={promptVariables}
          accessToken={accessToken}
          version={version?.replace("v", "") || "1"}
          proxySettings={proxySettings}
        />
        {editMode && onShowHistory && (
          <Button variant="secondary" onClick={onShowHistory}>
            <Clock className="h-4 w-4" />
            History
          </Button>
        )}
        <Button onClick={onSave} disabled={isSaving}>
          <Save className="h-4 w-4" />
          {isSaving ? "Saving…" : editMode ? "Update" : "Save"}
        </Button>
      </div>
    </div>
  );
};

export default PromptEditorHeader;
