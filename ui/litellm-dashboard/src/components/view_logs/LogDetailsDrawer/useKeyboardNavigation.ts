import { useEffect } from "react";
import { LogEntry } from "../columns";
import { KEY_ESCAPE, KEY_J_LOWER, KEY_J_UPPER, KEY_K_LOWER, KEY_K_UPPER } from "./constants";

interface UseKeyboardNavigationProps {
  isOpen: boolean;
  currentLog: LogEntry | null;
  allLogs: LogEntry[];
  onClose: () => void;
  onSelectLog?: (log: LogEntry) => void;
}

/**
 * Custom hook for keyboard navigation in the log details drawer.
 * Handles J/K for next/previous and Escape for close.
 *
 * Keyboard shortcuts:
 * - J: Navigate to previous log (up)
 * - K: Navigate to next log (down)
 * - Escape: Close drawer
 */
export function useKeyboardNavigation({
  isOpen,
  currentLog,
  allLogs,
  onClose,
  onSelectLog,
}: UseKeyboardNavigationProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input
      if (isUserTyping(e.target)) {
        return;
      }

      if (!isOpen) return;

      switch (e.key) {
        case KEY_ESCAPE:
          onClose();
          break;
        case KEY_J_LOWER:
        case KEY_J_UPPER:
          selectPreviousLog();
          break;
        case KEY_K_LOWER:
        case KEY_K_UPPER:
          selectNextLog();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentLog, allLogs]);

  const selectNextLog = () => {
    if (!currentLog || !allLogs.length || !onSelectLog) return;

    const currentIndex = allLogs.findIndex((l) => l.request_id === currentLog.request_id);
    if (currentIndex < allLogs.length - 1) {
      onSelectLog(allLogs[currentIndex + 1]);
    }
  };

  const selectPreviousLog = () => {
    if (!currentLog || !allLogs.length || !onSelectLog) return;

    const currentIndex = allLogs.findIndex((l) => l.request_id === currentLog.request_id);
    if (currentIndex > 0) {
      onSelectLog(allLogs[currentIndex - 1]);
    }
  };

  return {
    selectNextLog,
    selectPreviousLog,
  };
}

/**
 * Checks if the user is currently typing in an input field.
 * Used to prevent keyboard shortcuts from interfering with text input.
 */
function isUserTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
}
