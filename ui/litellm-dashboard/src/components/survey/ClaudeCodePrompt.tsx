import React from "react";
import { Code } from "lucide-react";
import { NudgePrompt } from "./NudgePrompt";
import { useTranslation } from "react-i18next";

interface ClaudeCodePromptProps {
  onOpen: () => void;
  onDismiss: () => void;
  isVisible: boolean;
}

export function ClaudeCodePrompt({ onOpen, onDismiss, isVisible }: ClaudeCodePromptProps) {
  const { t } = useTranslation();

  return (
    <NudgePrompt
      onOpen={onOpen}
      onDismiss={onDismiss}
      isVisible={isVisible}
      title={t("survey.claudeCodePrompt.title")}
      description={t("survey.claudeCodePrompt.description")}
      buttonText={t("survey.claudeCodePrompt.buttonText")}
      icon={Code}
      accentColor="#7c3aed"
      buttonStyle={{ backgroundColor: "#7c3aed", borderColor: "#7c3aed" }}
    />
  );
}
