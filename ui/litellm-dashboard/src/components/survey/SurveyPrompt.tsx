import React from "react";
import { MessageSquare, X } from "lucide-react";
import { Button } from "antd";

interface SurveyPromptProps {
  onOpen: () => void;
  onDismiss: () => void;
  isVisible: boolean;
}

export function SurveyPrompt({ onOpen, onDismiss, isVisible }: SurveyPromptProps) {
  if (!isVisible) return null;

  return (
    <div
      className={`fixed bottom-6 right-6 z-40 w-80 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden transform transition-all duration-300 ease-out ${
        isVisible ? "translate-y-0 opacity-100 scale-100" : "translate-y-4 opacity-0 scale-95"
      }`}
    >
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-2 text-blue-600">
            <MessageSquare className="h-5 w-5" />
            <span className="font-semibold text-sm">Quick feedback</span>
          </div>
          <button
            onClick={onDismiss}
            className="text-gray-400 hover:text-gray-600 transition-colors p-0.5 rounded hover:bg-gray-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="text-sm text-gray-600 mb-3">
          Help us improve LiteLLM! Share your experience in 3 quick questions.
        </p>

        <Button type="primary" block onClick={onOpen}>
          Share feedback
        </Button>
      </div>

      <div className="h-1 bg-gradient-to-r from-blue-500 to-blue-600" />
    </div>
  );
}

