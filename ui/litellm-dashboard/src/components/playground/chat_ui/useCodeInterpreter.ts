/**
 * Custom hook for managing Code Interpreter state.
 * Container creation is handled automatically by OpenAI with container: { type: "auto" }
 */

import { useState, useCallback } from "react";
import { CodeInterpreterResult } from "../llm_calls/code_interpreter_handler";

export interface UseCodeInterpreterReturn {
  // State
  enabled: boolean;
  result: CodeInterpreterResult | null;
  
  // Actions
  setEnabled: (enabled: boolean) => void;
  setResult: (result: CodeInterpreterResult | null) => void;
  clearResult: () => void;
  toggle: () => void;
}

export function useCodeInterpreter(): UseCodeInterpreterReturn {
  const [enabled, setEnabledState] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    const saved = sessionStorage.getItem("codeInterpreterEnabled");
    return saved ? JSON.parse(saved) : false;
  });

  const [result, setResult] = useState<CodeInterpreterResult | null>(null);

  // Persist enabled state to session storage
  const setEnabled = useCallback((value: boolean) => {
    setEnabledState(value);
    if (typeof window !== "undefined") {
      sessionStorage.setItem("codeInterpreterEnabled", JSON.stringify(value));
    }
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
  }, []);

  const toggle = useCallback(() => {
    setEnabled(!enabled);
  }, [enabled, setEnabled]);

  return {
    enabled,
    result,
    setEnabled,
    setResult,
    clearResult,
    toggle,
  };
}

// Re-export the type for convenience
export type { CodeInterpreterResult } from "../llm_calls/code_interpreter_handler";
