import React, { useEffect, useState } from "react";
import { X, LucideIcon, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import {
  setLocalStorageItem,
  emitLocalStorageChange,
} from "@/utils/localStorageUtils";

interface NudgePromptProps {
  onOpen: () => void;
  onDismiss: () => void;
  isVisible: boolean;
  title: string;
  description: string;
  buttonText: string;
  icon: LucideIcon;
  accentColor: string;
  buttonStyle?: React.CSSProperties;
}

const DISMISS_DURATION = 15000;
const CONFIRMATION_DURATION = 5000;

export function NudgePrompt({
  onOpen,
  onDismiss,
  isVisible,
  title,
  description,
  buttonText,
  icon: Icon,
  accentColor,
  buttonStyle,
}: NudgePromptProps) {
  const disableShowPrompts = useDisableShowPrompts();
  const [progress, setProgress] = useState(100);
  const [showConfirmation, setShowConfirmation] = useState(false);

  useEffect(() => {
    if (!isVisible) {
      setProgress(100);
      setShowConfirmation(false);
      return;
    }

    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(
        0,
        100 - (elapsed / DISMISS_DURATION) * 100,
      );
      setProgress(remaining);

      if (remaining <= 0) {
        clearInterval(interval);
      }
    }, 50);

    return () => clearInterval(interval);
  }, [isVisible]);

  useEffect(() => {
    if (showConfirmation) {
      const timer = setTimeout(() => {
        setShowConfirmation(false);
        onDismiss();
      }, CONFIRMATION_DURATION);

      return () => clearTimeout(timer);
    }
  }, [showConfirmation, onDismiss]);

  const handleDontAskAgain = () => {
    setLocalStorageItem("disableShowPrompts", "true");
    emitLocalStorageChange("disableShowPrompts");
    setShowConfirmation(true);
  };

  if (showConfirmation) {
    return (
      <div
        className={cn(
          "fixed bottom-6 right-6 z-40 w-80 bg-background rounded-lg shadow-xl border border-border overflow-hidden transform transition-all duration-300 ease-out",
          isVisible
            ? "translate-y-0 opacity-100 scale-100"
            : "translate-y-4 opacity-0 scale-95",
        )}
      >
        <div className="p-4">
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-100 dark:bg-emerald-950 flex items-center justify-center">
              <Check className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div className="flex-1">
              <p className="text-sm text-foreground font-medium">
                Got it, we will not ask again. Reactivate this at any time in
                the User Menu.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!isVisible || disableShowPrompts) return null;

  return (
    <div
      className={cn(
        "fixed bottom-6 right-6 z-40 w-80 bg-background rounded-lg shadow-xl border border-border overflow-hidden transform transition-all duration-300 ease-out",
        isVisible
          ? "translate-y-0 opacity-100 scale-100"
          : "translate-y-4 opacity-0 scale-95",
      )}
    >
      <div className="h-1 bg-muted w-full">
        <div
          className="h-full transition-all duration-100 ease-linear"
          style={{ width: `${progress}%`, backgroundColor: accentColor }}
        />
      </div>

      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div
            className="flex items-center gap-2"
            style={{ color: accentColor }}
          >
            <Icon className="h-5 w-5" />
            <span className="font-semibold text-sm">{title}</span>
          </div>
          <button
            onClick={onDismiss}
            className="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded hover:bg-muted"
            aria-label="Dismiss"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="text-sm text-muted-foreground mb-3">{description}</p>

        <div className="space-y-2">
          <Button className="w-full" onClick={onOpen} style={buttonStyle}>
            {buttonText}
          </Button>
          <Button
            variant="outline"
            className="w-full text-xs text-destructive border-destructive/30 hover:bg-destructive/10"
            onClick={handleDontAskAgain}
          >
            Don&apos;t ask me again
          </Button>
        </div>
      </div>
    </div>
  );
}
