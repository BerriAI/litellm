import React, { useEffect, useState } from "react";
import { X, LucideIcon } from "lucide-react";
import { Button } from "antd";

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
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    if (!isVisible) {
      setProgress(100);
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

  if (!isVisible) return null;

  return (
    <div
      className={`fixed bottom-6 right-6 z-40 w-80 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden transform transition-all duration-300 ease-out ${
        isVisible ? "translate-y-0 opacity-100 scale-100" : "translate-y-4 opacity-0 scale-95"
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

        <Button type="primary" block onClick={onOpen} style={buttonStyle}>
          {buttonText}
        </Button>
      </div>
    </div>
  );
}

