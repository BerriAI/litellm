import React from "react";
import { MessageSquare } from "lucide-react";
import { NudgePrompt } from "./NudgePrompt";
import { useTranslation } from "react-i18next";

interface SurveyPromptProps {
  onOpen: () => void;
  onDismiss: () => void;
  isVisible: boolean;
}

export function SurveyPrompt({ onOpen, onDismiss, isVisible }: SurveyPromptProps) {
  const { t } = useTranslation();

  return (
    <NudgePrompt
      onOpen={onOpen}
      onDismiss={onDismiss}
      isVisible={isVisible}
      title={t("survey.surveyPrompt.title")}
      description={t("survey.surveyPrompt.description")}
      buttonText={t("survey.surveyPrompt.buttonText")}
      icon={MessageSquare}
      accentColor="#3b82f6"
    />
  );
}
