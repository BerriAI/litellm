"use client";

import React, { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { decisionsTestCall } from "@/components/networking";
import {
  buildDecisionTestBody,
  decisionTestOutcome,
  parseDecisionTestResponse,
  type DecisionTestVerdict,
} from "./decisionModelQuestion";

export interface DecisionQuestionTestButtonProps {
  name: string;
  open: boolean;
  onToggle: () => void;
}

export const DecisionQuestionTestButton: React.FC<DecisionQuestionTestButtonProps> = ({ name, open, onToggle }) => (
  <button
    type="button"
    aria-label={`Test ${name}`}
    aria-expanded={open}
    onClick={onToggle}
    className="inline-flex items-center rounded-md border border-border px-2.5 py-1 text-xs font-medium hover:bg-accent"
  >
    Test
  </button>
);

export interface DecisionQuestionTestPanelProps {
  accessToken: string | null;
  model: string;
  name: string;
  instructions: string;
  action: "block" | "log";
  threshold: number;
}

const OUTCOME_LABELS: Record<string, string> = {
  would_block: "Would block",
  would_log: "Would log",
  pass: "Pass",
  refused: "Refused",
};

export const DecisionQuestionTestPanel: React.FC<DecisionQuestionTestPanelProps> = ({
  accessToken,
  model,
  name,
  instructions,
  action,
  threshold,
}) => {
  const [input, setInput] = useState("");
  const [verdict, setVerdict] = useState<DecisionTestVerdict | null>(null);
  const [loading, setLoading] = useState(false);

  const canTest = Boolean(model && input.trim());

  useEffect(() => {
    if (!canTest) {
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const body = buildDecisionTestBody(model, input, name, instructions);
        const data = await decisionsTestCall(accessToken ?? "", body, controller.signal);
        setVerdict(parseDecisionTestResponse(data, name));
      } catch (error) {
        if (controller.signal.aborted) return;
        setVerdict({ kind: "error", message: error instanceof Error ? error.message : String(error) });
      } finally {
        setLoading(false);
      }
    }, 500);
    return () => {
      clearTimeout(timer);
      controller.abort();
      setLoading(false);
    };
  }, [canTest, input, model, name, instructions, accessToken]);

  const currentVerdict = canTest ? verdict : null;
  const outcome = currentVerdict ? decisionTestOutcome(currentVerdict, action, threshold) : null;
  const probability = currentVerdict?.kind === "probability" ? currentVerdict.probability : null;
  const errorMessage = !loading && currentVerdict?.kind === "error" ? currentVerdict.message : null;
  const outcomeLabel = !loading && outcome !== null && outcome !== "error" ? OUTCOME_LABELS[outcome] : null;

  return (
    <div data-slot="decision-question-test-panel" className="mt-3 space-y-2">
      <Textarea
        aria-label={`Try an input for ${name}`}
        rows={2}
        placeholder="Try an input"
        className="w-full resize-none"
        value={input}
        onChange={(event) => setInput(event.target.value)}
      />
      {!model && <p className="m-0 text-xs text-muted-foreground">Select a decision model to test this question</p>}
      {model && !input.trim() && (
        <p className="m-0 text-xs text-muted-foreground">Type an input to test this question</p>
      )}
      {loading && <p className="m-0 text-xs text-muted-foreground">Testing…</p>}
      {errorMessage && <p className="m-0 text-xs text-destructive">{errorMessage}</p>}
      {outcomeLabel && (
        <div className="flex items-center gap-2 overflow-hidden whitespace-nowrap text-xs">
          {probability !== null && (
            <span className="text-muted-foreground">
              p={probability.toFixed(2)} vs threshold {threshold.toFixed(2)}
            </span>
          )}
          <Badge variant={outcome === "pass" ? "secondary" : "default"}>{outcomeLabel}</Badge>
        </div>
      )}
    </div>
  );
};
