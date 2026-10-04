import { ArrowRight, Check, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TracingSetupFields } from "@/components/view_logs/TraceView/TracingSetupCard";
import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";
import type { LensSetupState } from "./useLensSetup";

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
  if (!state.tracingEnabled) return 0;
  if (!state.tracesReady) return 1;
  return state.connected ? 3 : 2;
}

function StorageStep({ state, onStep, ...props }: StepProps) {
  if (!state.tracingEnabled)
    return (
      <TracingSetupFields
        detail="Tracing is not enabled"
        accessToken={props.accessToken}
        onOpenTrace={props.onTrace}
        onCheck={state.refresh}
        checking={state.checking}
        readOnly={props.readOnly}
      />
    );
  return (
    <div className="space-y-4">
      <p role="status" className="text-sm text-success">
        Trace storage is connected
      </p>
      <Button onClick={() => onStep(1)}>
        Continue to your agent <ArrowRight aria-hidden="true" className="size-4" />
      </Button>
    </div>
  );
}

function continuationLabel(state: LensSetupState) {
  if (state.connected) return "Continue to investigation";
  return state.tracesReady ? "Continue to worker" : "Continue with request logs";
}

function AgentStep({ state, onConnect, onCreate, ...props }: StepProps) {
  if (!state.tracingEnabled)
    return <p className="text-sm text-muted-foreground">Connect trace storage in step 1 before sending a trace.</p>;
  const activityReady = state.tracesReady || state.requestsReady;
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
      {activityReady && (
        <div className="mt-4">
          <Button onClick={state.connected ? onCreate : onConnect} disabled={props.readOnly || !props.canInvestigate}>
            {continuationLabel(state)}
            <ArrowRight aria-hidden="true" className="size-4" />
          </Button>
          {state.requestsReady && !state.tracesReady && (
            <p className="mt-2 text-xs text-muted-foreground">
              Request logs are already available. You can investigate them now and add agent traces later.
            </p>
          )}
        </div>
      )}
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
    <ol className="divide-y overflow-hidden rounded-lg border">
      {items.map((item, index) => (
        <li key={item.title}>
          <h3>
            <button
              type="button"
              aria-expanded={step === index}
              aria-controls={`lens-setup-step-${index}`}
              onClick={() => props.onStep(index)}
              className="flex w-full items-start gap-3 p-5 text-left hover:bg-muted/30"
            >
              <span
                className={
                  item.complete
                    ? "flex size-7 shrink-0 items-center justify-center rounded-full bg-success/10 text-success"
                    : "flex size-7 shrink-0 items-center justify-center rounded-full border text-xs font-medium"
                }
                aria-label={item.complete ? `Step ${index + 1} complete` : `Step ${index + 1}`}
              >
                {item.complete ? <Check aria-hidden="true" className="size-4" /> : index + 1}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium">{item.title}</span>
                <span className="mt-1 block text-sm font-normal text-muted-foreground">{item.description}</span>
              </span>
              <ChevronDown
                aria-hidden="true"
                className={`mt-1 size-4 shrink-0 text-muted-foreground ${step === index ? "rotate-180" : ""}`}
              />
            </button>
          </h3>
          <div id={`lens-setup-step-${index}`} hidden={step !== index} className="px-5 pb-5 sm:pl-15">
            {item.content}
          </div>
        </li>
      ))}
    </ol>
  );
}
