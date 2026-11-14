"use client";

import React, { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { StatusBadge } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";
import { FieldLabel } from "@/components/ui/field";
import { Textarea } from "@/components/ui/textarea";
import { decisionsTestCall } from "@/components/networking";
import type { DecisionModelCheckDraft } from "./buildDecisionModelParams";
import {
  buildDecisionTestBody,
  DECISION_TEST_CHIP_TONE,
  decisionTestChip,
  decisionTestOverall,
  parseDecisionTestResponse,
  prependTestRun,
  runnableDecisionQuestions,
  runTestShortcutLabel,
  visibleTestResults,
  type DecisionTestRun,
} from "./decisionModelQuestion";

export interface DecisionTestSectionProps {
  accessToken: string | null;
  model: string;
  checks: DecisionModelCheckDraft[];
}

const DecisionTestSection: React.FC<DecisionTestSectionProps> = ({ accessToken, model, checks }) => {
  const { t } = useTranslation();
  const [input, setInput] = useState("");
  const [runs, setRuns] = useState<DecisionTestRun[]>([]);
  const [running, setRunning] = useState(false);
  const nextRunId = useRef(1);
  const controllerRef = useRef<AbortController | null>(null);

  const runnableQuestions = runnableDecisionQuestions(checks);
  let disabledReason: string | null = null;
  if (!model) disabledReason = t("guardrails.decisionModel.test.disabled.noModel");
  else if (!input.trim()) disabledReason = t("guardrails.decisionModel.test.disabled.noInput");
  else if (runnableQuestions.length === 0) disabledReason = t("guardrails.decisionModel.test.disabled.noQuestions");
  const canRun = !disabledReason && !running;

  const runTest = async () => {
    if (!canRun) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const asked = Object.fromEntries(runnableQuestions.map((check) => [check.name, check.instructions]));
    const id = nextRunId.current++;
    setRunning(true);
    try {
      const data = await decisionsTestCall(
        accessToken ?? "",
        buildDecisionTestBody(model, input, checks),
        controller.signal,
      );
      const run: DecisionTestRun = {
        id,
        input,
        model,
        asked,
        results: parseDecisionTestResponse(data, Object.keys(asked)),
      };
      setRuns((prev) => prependTestRun(prev, run));
    } catch (error) {
      if (controller.signal.aborted) return;
      setRuns((prev) =>
        prependTestRun(prev, {
          id,
          input,
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      if (controllerRef.current === controller) setRunning(false);
    }
  };

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div className="space-y-1">
        <FieldLabel>{t("guardrails.decisionModel.test.label")}</FieldLabel>
        <p className="m-0 text-xs text-muted-foreground">{t("guardrails.decisionModel.test.description")}</p>
      </div>
      <Textarea
        aria-label={t("guardrails.decisionModel.test.inputAriaLabel")}
        rows={3}
        placeholder={t("guardrails.decisionModel.test.inputPlaceholder")}
        className="w-full resize-none"
        value={input}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            runTest();
          }
        }}
      />
      <div className="flex items-center gap-2">
        <Button type="button" onClick={runTest} disabled={!canRun}>
          {running ? t("guardrails.decisionModel.test.running") : t("guardrails.decisionModel.test.run")}
        </Button>
        {!running && disabledReason && <p className="m-0 text-xs text-muted-foreground">{disabledReason}</p>}
        {!running && !disabledReason && <p className="m-0 text-xs text-muted-foreground">{runTestShortcutLabel()}</p>}
      </div>
      {runs.length > 0 && (
        <>
          <p className="m-0 text-xs text-muted-foreground">{t("guardrails.decisionModel.test.historyHint")}</p>
          <div className="space-y-2" data-slot="decision-test-history">
            {runs.map((run) => {
              const visible = "error" in run ? [] : visibleTestResults(run, checks, model);
              return (
                <div key={run.id} className="rounded-md border border-border px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="m-0 min-w-0 truncate text-xs font-medium text-foreground" title={run.input}>
                      {run.input}
                    </p>
                    {visible.length > 0 && (
                      <StatusBadge
                        tone={DECISION_TEST_CHIP_TONE[decisionTestOverall(visible)]}
                        label={
                          decisionTestOverall(visible) === "block"
                            ? t("guardrails.decisionModel.test.overall.block")
                            : t("guardrails.decisionModel.test.overall.pass")
                        }
                      />
                    )}
                  </div>
                  {"error" in run && <p className="m-0 mt-1.5 text-xs text-destructive">{run.error}</p>}
                  {!("error" in run) && visible.length === 0 && (
                    <p className="m-0 mt-1.5 text-xs text-muted-foreground">
                      {t("guardrails.decisionModel.test.stale")}
                    </p>
                  )}
                  {visible.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {visible.map(({ check, result }) => {
                        const chip = decisionTestChip(result, check);
                        const score =
                          result.kind === "probability"
                            ? result.probability.toFixed(2)
                            : t("guardrails.decisionModel.test.chip.noAnswer");
                        let label = "";
                        if (chip === "block") label = t("guardrails.decisionModel.test.chip.block");
                        else if (chip === "logged") label = t("guardrails.decisionModel.test.chip.logged");
                        else if (chip === "pass") label = t("guardrails.decisionModel.test.chip.pass");
                        return (
                          <StatusBadge
                            key={check.name}
                            tone={DECISION_TEST_CHIP_TONE[chip]}
                            label={label ? `${check.name} ${score} ${label}` : `${check.name} ${score}`}
                          />
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};

export default DecisionTestSection;
