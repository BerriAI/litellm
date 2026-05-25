import { useEffect } from "react";
import { LogEntry } from "../columns";
import { KEY_ESCAPE, KEY_J_LOWER, KEY_J_UPPER, KEY_K_LOWER, KEY_K_UPPER } from "./constants";

interface UseKeyboardNavigationProps {
  isOpen: boolean;
  currentLog: LogEntry | null;
  allLogs: LogEntry[];
  onClose: () => void;
  onSelectLog?: (log: LogEntry) => void;
  onPreviousPage?: () => void;
  onNextPage?: () => void;
}

/**
 * Custom hook for keyboard navigation in the log details drawer.
 *
 * Keyboard shortcuts:
 * - J / K: Navigate to next / previous log within the current page
 * - Shift+J / Shift+K: Page next / previous (session mode)
 * - Escape: Close drawer
 */
export function useKeyboardNavigation({
  isOpen,
  currentLog,
  allLogs,
  onClose,
  onSelectLog,
  onPreviousPage,
  onNextPage,
}: UseKeyboardNavigationProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input
      if (isUserTyping(e.target)) {
        return;
      }

      if (!isOpen) return;

      if (e.key === KEY_ESCAPE) {
        onClose();
        return;
      }

      // With Shift held, e.key is already uppercase ("J"/"K"). Compare against
      // both cases so the page shortcuts work regardless of caps lock.
      const isJ = e.key === KEY_J_LOWER || e.key === KEY_J_UPPER;
      const isK = e.key === KEY_K_LOWER || e.key === KEY_K_UPPER;
      if (!isJ && !isK) return;

      if (e.shiftKey) {
        if (isJ) onNextPage?.();
        else onPreviousPage?.();
      } else {
        if (isJ) selectNextLog();
        else selectPreviousLog();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentLog, allLogs, onClose, onSelectLog, onPreviousPage, onNextPage]);

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
