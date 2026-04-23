import React from "react";
import { X, Code, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ClaudeCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
}

const GOOGLE_FORM_URL = "https://forms.gle/LZeJQ3XytBakckYa9";

export function ClaudeCodeModal({
  isOpen,
  onClose,
  onComplete,
}: ClaudeCodeModalProps) {
  if (!isOpen) return null;

  const handleOpenForm = () => {
    window.open(GOOGLE_FORM_URL, "_blank", "noopener,noreferrer");
    onComplete();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      <div
        className="fixed inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative w-full max-w-md bg-background rounded-xl shadow-2xl overflow-hidden transform transition-all duration-300 ease-out">
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-muted/50">
          <div className="flex items-center gap-2 text-purple-600 dark:text-purple-400">
            <Code className="h-5 w-5" />
            <span className="font-semibold text-sm tracking-wide uppercase">
              Claude Code Feedback
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded-full hover:bg-accent"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="p-8">
          <h2 className="text-2xl font-bold text-foreground mb-4">
            Help us improve your experience
          </h2>
          <p className="text-muted-foreground mb-6">
            We&apos;d love to hear about your experience using LiteLLM with
            Claude Code. Your feedback helps us improve the product for
            everyone.
          </p>
          <p className="text-sm text-muted-foreground mb-6">
            This brief survey takes about 2-3 minutes to complete.
          </p>

          <Button
            size="lg"
            className="w-full bg-purple-600 hover:bg-purple-700 text-white"
            onClick={handleOpenForm}
          >
            <ExternalLink className="h-4 w-4" />
            Open Feedback Form
          </Button>
        </div>
      </div>
    </div>
  );
}
