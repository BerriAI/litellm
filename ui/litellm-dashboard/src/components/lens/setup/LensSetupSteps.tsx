import { ArrowRight, Check, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TracingSetupFields } from "@/components/view_logs/TraceView/TracingSetupCard";
import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";
import type { LensSetupState } from "./useLensSetup";
import { cn } from "@/lib/cva.config";

type StepProps = {
  accessToken: string;
  state: LensSetupState;
  readOnly: boolean;
  canInvestigate: boolean;
  canMintTracingKey: boolean;
  onStep: (step: number) => void;
  onTrace: (trace: TraceSummary) => void;
  onConnect: () => void;
  onCreate: () => void;
};

export function initialSetupStep(state: LensSetupState) {
  if (state.tracesReady || state.requestsReady) return state.connected ? 3 : 2;
  return state.tracingEnabled ? 1 : 0;
}

function StorageStep({ state, onStep, ...props }: StepProps) {
  if (!state.tracingEnabled)
    return (
      <>
        <TracingSetupFields
          detail="Tracing is not enabled"
          accessToken={props.accessToken}
          onOpenTrace={props.onTrace}
          onCheck={state.refresh}
          checking={state.checking}
          readOnly={props.readOnly}
        />
        <ActivityContinuation state={state} {...props} />
      </>
    );
  return (
    <div className="space-y-4">
      <p role="status" className="text-sm text-success">
        Trace storage is connected
      </p>
      <Button onClick={() => onStep(1)}>
        Continue to your agent <ArrowRight aria-hidden="true" className="size-4" />
      </Button>
      <ActivityContinuation state={state} {...props} />
    </div>
  );
}

function continuationLabel(state: LensSetupState) {
  if (state.connected) return "Continue to investigation";
  return state.tracesReady ? "Continue to worker" : "Continue with request logs";
}

function ActivityContinuation({
  state,
  onConnect,
  onCreate,
  readOnly,
  canInvestigate,
}: Pick<StepProps, "state" | "onConnect" | "onCreate" | "readOnly" | "canInvestigate">) {
  if (!state.tracesReady && !state.requestsReady) return null;
  return (
    <div className="mt-4">
      <Button onClick={state.connected ? onCreate : onConnect} disabled={readOnly || !canInvestigate}>
        {continuationLabel(state)}
        <ArrowRight aria-hidden="true" className="size-4" />
      </Button>
      {state.requestsReady && !state.tracesReady && (
        <p className="mt-2 text-xs text-muted-foreground">
          Request logs are already available. You can investigate them now and add agent traces later.
        </p>
      )}
    </div>
  );
}

function AgentStep({ state, ...props }: StepProps) {
  if (!state.tracingEnabled)
    return (
      <>
        <p className="text-sm text-muted-foreground">Connect trace storage in step 1 before sending a trace.</p>
        <ActivityContinuation state={state} {...props} />
      </>
    );
  return (
    <>
      <div hidden={state.tracesReady}>
        <TracingSetupFields
          detail={null}
          accessToken={props.accessToken}
          onOpenTrace={(value) => {
            state.refresh();
            props.onTrace(value);
          }}
          onCheck={state.refresh}
          checking={state.checking}
          readOnly={props.readOnly}
          canMintTracingKey={props.canMintTracingKey}
        />
      </div>
      {state.tracesReady && (
        <p role="status" className="text-sm text-success">
          Your first trace is ready. Continue setup so Lens can investigate your agent’s behavior.
        </p>
      )}
      <ActivityContinuation state={state} {...props} />
    </>
  );
}

function WorkerStep({ state, onConnect, onCreate, readOnly, canInvestigate }: StepProps) {
  const activityReady = state.tracesReady || state.requestsReady;
  return (
    <div className="space-y-4">
      <p role="status" className="text-sm text-muted-foreground">
        {state.connected
          ? "Worker connected. You’re ready to create an investigation."
          : "The worker reviews recorded activity using a model on your gateway. You choose its analysis model and spending limit."}
      </p>
      {!activityReady && (
        <p className="text-sm text-muted-foreground">Record activity in step 2 before connecting a worker.</p>
      )}
      <Button onClick={state.connected ? onCreate : onConnect} disabled={!activityReady || readOnly || !canInvestigate}>
        {state.connected ? "Continue to investigation" : "Connect worker"}
        <ArrowRight aria-hidden="true" className="size-4" />
      </Button>
    </div>
  );
}

function InvestigationStep({ state, onCreate, readOnly, canInvestigate }: StepProps) {
  return (
    <div className="space-y-4">
      <p className="text-sm leading-6 text-muted-foreground">
        Choose the activity to review and describe how your agent should behave. Lens will show findings with evidence
        and suggested changes.
      </p>
      {!state.ready && (
        <p className="text-sm text-muted-foreground">
          Recorded activity and a connected worker are required before you can run an investigation.
        </p>
      )}
      <Button onClick={onCreate} disabled={!state.ready || readOnly || !canInvestigate}>
        New investigation <ArrowRight aria-hidden="true" className="size-4" />
      </Button>
    </div>
  );
}

export function LensSetupSteps({ step, ...props }: StepProps & { step: number }) {
  const items = [
    {
      title: "Enable tracing on the gateway",
      description: "Connect ClickHouse and restart the gateway.",
      complete: props.state.tracingEnabled,
      content: <StorageStep {...props} />,
    },
    {
      title: "Send your first trace",
      description: "Capture your agent’s inputs, outputs, and tool calls.",
      complete: props.state.tracesReady,
      content: <AgentStep {...props} />,
    },
    {
      title: "Connect a worker",
      description: "Choose a model and run the worker on your infrastructure.",
      complete: props.state.connected,
      content: <WorkerStep {...props} />,
    },
    {
      title: "Run your first investigation",
      description: "Describe the expected behavior and review a sample of activity.",
      complete: props.state.hasInvestigations,
      content: <InvestigationStep {...props} />,
    },
  ];
  return (
    <ol className="divide-y overflow-hidden rounded-2xl border bg-card">
      {items.map((item, index) => (
        <li key={item.title}>
          <h3>
            <button
              type="button"
              aria-expanded={step === index}
              aria-controls={`lens-setup-step-${index}`}
              onClick={() => props.onStep(index)}
              className="group flex w-full items-start gap-4 p-5 text-left outline-none hover:bg-muted/30 focus-visible:bg-muted/50 sm:p-6"
            >
              <span
                className={cn(
                  "flex size-8 shrink-0 items-center justify-center rounded-full text-sm font-medium",
                  item.complete ? "bg-success/10 text-success" : "border text-muted-foreground",
                  !item.complete && step === index && "border-primary bg-primary text-primary-foreground",
                )}
                aria-label={item.complete ? `Step ${index + 1} complete` : `Step ${index + 1}`}
              >
                {item.complete ? <Check aria-hidden="true" className="size-4" /> : index + 1}
              </span>
              <span className="min-w-0 flex-1">
                <span
                  className={cn(
                    "block text-sm font-medium sm:text-base",
                    step !== index && "text-muted-foreground group-hover:text-foreground",
                  )}
                >
                  {item.title}
                </span>
                <span className="mt-1 block text-sm leading-6 font-normal text-muted-foreground">
                  {item.description}
                </span>
              </span>
              <ChevronDown
                aria-hidden="true"
                className={`mt-1 size-4 shrink-0 text-muted-foreground ${step === index ? "rotate-180" : ""}`}
              />
            </button>
          </h3>
          <div id={`lens-setup-step-${index}`} hidden={step !== index} className="px-5 pb-6 sm:pr-6 sm:pb-7 sm:pl-18">
            {item.content}
          </div>
        </li>
      ))}
    </ol>
  );
}
