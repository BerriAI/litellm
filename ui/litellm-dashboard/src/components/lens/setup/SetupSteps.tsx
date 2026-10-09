"use client";

import { createContext, useContext, type ComponentProps } from "react";
import { cn } from "@/lib/cva.config";
import { StepIndicator, type StepState } from "../ui/StepIndicator";
import { SETUP_STEPS, type SetupStep as SetupStepId } from "./investigationSchema";

interface StepsContext {
  readonly current: SetupStepId;
  readonly onOpen: (step: SetupStepId) => void;
}

const Steps = createContext<StepsContext | null>(null);

function useSteps(): StepsContext {
  const context = useContext(Steps);
  if (!context) throw new Error("SetupStep must be rendered inside SetupSteps");
  return context;
}

export function nextSetupStep(step: SetupStepId): SetupStepId | undefined {
  return SETUP_STEPS[SETUP_STEPS.indexOf(step) + 1];
}

function stepState(distance: number): StepState {
  if (distance === 0) return "current";
  return distance < 0 ? "complete" : "upcoming";
}

export type SetupStepsProps = ComponentProps<"ol"> & {
  current: SetupStepId;
  onOpen: (step: SetupStepId) => void;
};

/** Vertical stepper: finished steps collapse to a summary and reopen on click, the current one shows its fields. */
export function SetupSteps({ current, onOpen, className, ...props }: SetupStepsProps) {
  return (
    <Steps.Provider value={{ current, onOpen }}>
      <ol data-slot="setup-steps" className={cn("min-w-0", className)} {...props} />
    </Steps.Provider>
  );
}

export type SetupStepProps = ComponentProps<"li"> & {
  id: SetupStepId;
  heading: string;
  description: string;
  summary: string;
};

export function SetupStep({ id, heading, description, summary, className, children, ...props }: SetupStepProps) {
  const { current, onOpen } = useSteps();
  const position = SETUP_STEPS.indexOf(id);
  const state = stepState(position - SETUP_STEPS.indexOf(current));
  return (
    <li
      data-slot="setup-step"
      data-state={state}
      className={cn("group relative flex gap-4 [&:not(:last-child)]:pb-8", className)}
      {...props}
    >
      <span aria-hidden="true" className="absolute top-7 bottom-0 left-3.5 w-px bg-border group-last:hidden" />
      <StepIndicator aria-hidden="true" index={position} state={state} />
      <div className="min-w-0 flex-1 pt-0.5">
        <button
          type="button"
          disabled={state !== "complete"}
          aria-current={state === "current" ? "step" : undefined}
          onClick={() => onOpen(id)}
          className="flex w-full flex-col items-start gap-0.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:cursor-default"
        >
          <span className="text-sm font-semibold group-data-[state=upcoming]:text-muted-foreground">{heading}</span>
          <span className="text-xs text-muted-foreground">{state === "complete" ? summary : description}</span>
        </button>
        {state === "current" && <div className="mt-4 space-y-5">{children}</div>}
      </div>
    </li>
  );
}
