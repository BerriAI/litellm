/**
 * Custom hook for managing Code Interpreter state and logic.
 */

import { useState, useCallback } from "react";
import { CodeInterpreterResult } from "../llm_calls/code_interpreter_handler";

export interface UseCodeInterpreterReturn {
  // State
  enabled: boolean;
  result: CodeInterpreterResult | null;
  containerId: string | null;
  
  // Actions
  setEnabled: (enabled: boolean) => void;
  setResult: (result: CodeInterpreterResult | null) => void;
  setContainerId: (containerId: string | null) => void;
  clearResult: () => void;
  toggle: () => void;
}

export function useCodeInterpreter(): UseCodeInterpreterReturn {
  const [enabled, setEnabled] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    const saved = sessionStorage.getItem("codeInterpreterEnabled");
    return saved ? JSON.parse(saved) : false;
  });

  const [result, setResult] = useState<CodeInterpreterResult | null>(null);

  const [containerId, setContainerId] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    return sessionStorage.getItem("selectedContainerId") || null;
  });

  // Persist enabled state to session storage
  const handleSetEnabled = useCallback((value: boolean) => {
    setEnabled(value);
    if (typeof window !== "undefined") {
      sessionStorage.setItem("codeInterpreterEnabled", JSON.stringify(value));
    }
  }, []);

  // Persist container ID to session storage
  const handleSetContainerId = useCallback((value: string | null) => {
    setContainerId(value);
    if (typeof window !== "undefined") {
      if (value) {
        sessionStorage.setItem("selectedContainerId", value);
      } else {
        sessionStorage.removeItem("selectedContainerId");
      }
    }
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
  }, []);

  const toggle = useCallback(() => {
    handleSetEnabled(!enabled);
  }, [enabled, handleSetEnabled]);

  return {
    enabled,
    result,
    containerId,
    setEnabled: handleSetEnabled,
    setResult,
    setContainerId: handleSetContainerId,
    clearResult,
    toggle,
  };
}

// Re-export the type for convenience
export type { CodeInterpreterResult } from "../llm_calls/code_interpreter_handler";

