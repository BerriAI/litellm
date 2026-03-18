import React from "react";
import { Code } from "lucide-react";
import { NudgePrompt } from "./NudgePrompt";

interface ClaudeCodePromptProps {
  onOpen: () => void;
  onDismiss: () => void;
  isVisible: boolean;
}

export function ClaudeCodePrompt({ onOpen, onDismiss, isVisible }: ClaudeCodePromptProps) {
  return (
    <NudgePrompt
      onOpen={onOpen}
      onDismiss={onDismiss}
      isVisible={isVisible}
      title="Claude Code Feedback"
      description="Help us improve your Claude Code experience with LiteLLM! Share your feedback in 4 quick questions."
      buttonText="Share feedback"
      icon={Code}
      accentColor="#7c3aed"
      buttonStyle={{ backgroundColor: '#7c3aed', borderColor: '#7c3aed' }}
    />
  );
}

