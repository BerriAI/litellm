import React from "react";
import { useTranslation } from "react-i18next";
import { PromptType } from "./types";
import { convertToDotPrompt } from "./utils";

interface DotpromptViewTabProps {
  prompt: PromptType;
}

const DotpromptViewTab: React.FC<DotpromptViewTabProps> = ({ prompt }) => {
  const { t } = useTranslation();
  const dotpromptContent = convertToDotPrompt(prompt);

  return (
    <div className="p-6">
      <div className="mb-4">
        <h3 className="text-sm font-medium text-foreground mb-2">{t("prompts.dotpromptView.title")}</h3>
        <p className="text-xs text-muted-foreground">{t("prompts.dotpromptView.description")}</p>
      </div>
      <div className="bg-muted border border-border rounded-lg p-4 overflow-auto">
        <pre className="text-sm text-foreground font-mono whitespace-pre-wrap">{dotpromptContent}</pre>
      </div>
    </div>
  );
};

export default DotpromptViewTab;
