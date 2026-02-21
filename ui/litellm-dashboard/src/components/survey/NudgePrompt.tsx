import React, { useEffect, useState } from "react";
import { X, LucideIcon, Check } from "lucide-react";
import { Button } from "antd";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { setLocalStorageItem, emitLocalStorageChange } from "@/utils/localStorageUtils";

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

const DISMISS_DURATION = 15000; // 15 seconds
const CONFIRMATION_DURATION = 5000; // 5 seconds

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
      const remaining = Math.max(0, 100 - (elapsed / DISMISS_DURATION) * 100);
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

  // Show confirmation even if disableShowPrompts is true (since we just set it)
  if (showConfirmation) {
    return (
      <div
        className={`fixed bottom-6 right-6 z-40 w-80 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden transform transition-all duration-300 ease-out ${isVisible ? "translate-y-0 opacity-100 scale-100" : "translate-y-4 opacity-0 scale-95"
          }`}
      >
        <div className="p-4">
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
              <Check className="h-5 w-5 text-green-600" />
            </div>
            <div className="flex-1">
              <p className="text-sm text-gray-700 font-medium">
                Got it, we will not ask again. Reactivate this at any time in the User Menu.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Don't show the prompt if disabled (unless we're showing confirmation)
  if (!isVisible || disableShowPrompts) return null;

  return (
    <div
      className={`fixed bottom-6 right-6 z-40 w-80 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden transform transition-all duration-300 ease-out ${isVisible ? "translate-y-0 opacity-100 scale-100" : "translate-y-4 opacity-0 scale-95"
        }`}
    >
      {/* Progress bar at top showing time remaining */}
      <div className="h-1 bg-gray-100 w-full">
        <div
          className="h-full transition-all duration-100 ease-linear"
          style={{ width: `${progress}%`, backgroundColor: accentColor }}
        />
      </div>

      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-2" style={{ color: accentColor }}>
            <Icon className="h-5 w-5" />
            <span className="font-semibold text-sm">{title}</span>
          </div>
          <button
            onClick={onDismiss}
            className="text-gray-400 hover:text-gray-600 transition-colors p-0.5 rounded hover:bg-gray-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="text-sm text-gray-600 mb-3">{description}</p>

        <div className="space-y-2">
          <Button type="primary" block onClick={onOpen} style={buttonStyle}>
            {buttonText}
          </Button>
          <Button
            variant="outlined"
            danger
            block
            onClick={handleDontAskAgain}
            className="text-xs"
          >
            Don&apos;t ask me again
          </Button>
        </div>
      </div>
    </div>
  );
}

