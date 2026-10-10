import { Badge } from "@/components/ui/badge";
import { DEFAULT_DECISION_THRESHOLD } from "./decisionModelQuestion";

interface DecisionModelCheckView {
  name: string;
  action?: string;
  threshold?: number;
  instructions?: string;
}

interface DecisionModelParamsView {
  guardrail?: string;
  decision_model?: string;
  checks?: DecisionModelCheckView[];
}

const DecisionModelDetails = ({ litellmParams }: { litellmParams?: DecisionModelParamsView | null }) => {
  if (litellmParams?.guardrail !== "decision_model" || !litellmParams.decision_model) return null;
  return <DecisionModelSection decisionModel={litellmParams.decision_model} checks={litellmParams.checks ?? []} />;
};

const DecisionModelSection = ({
  decisionModel,
  checks,
}: {
  decisionModel: string;
  checks: DecisionModelCheckView[];
}) => (
  <div>
    <p className="font-medium">Decision Model</p>
    <div className="font-mono">{decisionModel}</div>
    <p className="mt-2 font-medium">Checks</p>
    <div className="mt-2 overflow-hidden rounded-lg border border-border">
      <div className="flex border-b border-border bg-muted/40 px-4 py-2 text-xs font-semibold">
        <span className="flex-1">Check</span>
        <span className="w-24 text-right">Action</span>
        <span className="w-24 text-right">Threshold</span>
      </div>
      {checks.map((check) => (
        <div key={check.name} className="flex items-center border-b border-border px-4 py-2 text-sm last:border-b-0">
          <span className="flex-1">{check.name.replace(/_/g, " ")}</span>
          <span className="w-24 text-right">
            <Badge variant="secondary">{check.action ?? "block"}</Badge>
          </span>
          <span className="w-24 text-right">{check.threshold ?? DEFAULT_DECISION_THRESHOLD}</span>
        </div>
      ))}
    </div>
  </div>
);

export default DecisionModelDetails;
