import React from "react";
import { t } from "@/i18n";
import { MessageSquare } from "lucide-react";
import { NudgePrompt } from "./NudgePrompt";

interface SurveyPromptProps {
  onOpen: () => void;
  onDismiss: () => void;
  isVisible: boolean;
}

export function SurveyPrompt({ onOpen, onDismiss, isVisible }: SurveyPromptProps) {
  return (
    <NudgePrompt
      onOpen={onOpen}
      onDismiss={onDismiss}
      isVisible={isVisible}
      title={t("common.quick_feedback")}
      description="Help us improve LiteLLM! Share your experience in 5 quick questions."
      buttonText="Share feedback"
      icon={MessageSquare}
      accentColor="#3b82f6"
    />
  );
}
