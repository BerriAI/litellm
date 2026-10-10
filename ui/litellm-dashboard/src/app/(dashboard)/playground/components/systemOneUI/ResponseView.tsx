import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, LoaderCircle } from "lucide-react";
import { useState } from "react";
import type { SystemOneAnswer, SystemOneResponse } from "./lib/schemas";

const SCORE_FORMAT = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

interface ResponseViewProps {
  response?: SystemOneResponse;
  fallbackModel?: string;
  latencyMs?: number;
  error?: string;
  isLoading: boolean;
}

function ProbabilityMeter({
  label,
  probability,
  selected = false,
}: {
  label: string;
  probability: number;
  selected?: boolean;
}) {
  const percentage = Math.round(probability * 100);
  return (
    <div className="grid gap-1.5">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className={selected ? "font-semibold" : "text-muted-foreground"}>{label}</span>
        <span className={selected ? "font-semibold tabular-nums" : "tabular-nums"}>{percentage}%</span>
      </div>
      <div
        role="meter"
        aria-label={`${label} probability`}
        aria-valuenow={percentage}
        aria-valuetext={selected ? `${percentage}%, selected` : `${percentage}%`}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-2 overflow-hidden rounded-full bg-muted"
      >
        <div
          className={`h-full rounded-full ${selected ? "bg-primary" : "bg-primary/40"}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function AnswerDetails({ answer }: { answer: SystemOneAnswer }) {
  if (answer.type === "noul") {
    const percentage = Math.round(answer.noul * 100);
    return (
      <div className="grid gap-3">
        <p className="text-sm font-medium">{percentage}% yes</p>
        <ProbabilityMeter label="Yes" probability={answer.noul} />
      </div>
    );
  }

  if (answer.type === "choice") {
    const options = Object.entries(answer.probabilities).sort(([, first], [, second]) => second - first);
    return (
      <div className="grid gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">Selected choice</span>
          <Badge>{answer.choice}</Badge>
          {answer.confidence !== undefined && (
            <Badge variant="outline">{Math.round(answer.confidence * 100)}% confidence</Badge>
          )}
        </div>
        <div className="grid gap-3">
          {options.map(([label, probability]) => (
            <ProbabilityMeter key={label} label={label} probability={probability} selected={label === answer.choice} />
          ))}
        </div>
      </div>
    );
  }

  const levels = Object.entries(answer.probabilities).sort(([first], [second]) => Number(first) - Number(second));
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">Score</span>
        <Badge>{SCORE_FORMAT.format(answer.score)}</Badge>
        {answer.confidence !== undefined && (
          <Badge variant="outline">{Math.round(answer.confidence * 100)}% confidence</Badge>
        )}
      </div>
      <div className="grid gap-3">
        {levels.map(([level, probability]) => {
          const description = answer.legend?.[level];
          const label =
            description === undefined
              ? level
              : `${level}: ${typeof description === "string" ? description : JSON.stringify(description)}`;
          return (
            <ProbabilityMeter
              key={level}
              label={label}
              probability={probability}
              selected={Number(level) === Math.round(answer.score)}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function ResponseView({ response, fallbackModel, latencyMs, error, isLoading }: ResponseViewProps) {
  const [showRaw, setShowRaw] = useState(false);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground">
          <LoaderCircle className="size-4 animate-spin" />
          Calculating calibrated probabilities
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>System One request failed</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!response) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Calibrated probabilities</CardTitle>
          <CardDescription>Send a request to see System One answers and probabilities.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const model = response.model ?? fallbackModel;

  return (
    <Card className="wrap-anywhere">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="grid gap-1">
            <CardTitle>Calibrated probabilities</CardTitle>
            {model && <CardDescription>{model}</CardDescription>}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {latencyMs !== undefined && <Badge variant="outline">{Math.round(latencyMs)} ms</Badge>}
            {response.usage && (
              <Badge variant="outline">
                {response.usage.input_tokens} input / {response.usage.output_tokens} output tokens
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid gap-3">
        {Object.entries(response.answers).map(([id, answer]) => (
          <Card key={id} size="sm">
            <CardHeader>
              <CardTitle className="font-mono">{id}</CardTitle>
              <CardAction>
                <Badge variant="secondary">{answer.type}</Badge>
              </CardAction>
            </CardHeader>
            <CardContent>
              <AnswerDetails answer={answer} />
            </CardContent>
          </Card>
        ))}
        <Collapsible open={showRaw} onOpenChange={setShowRaw}>
          <CollapsibleTrigger
            render={
              <Button variant="ghost" size="sm">
                <ChevronDown className="size-4" />
                Raw response
              </Button>
            }
          />
          <CollapsibleContent>
            <pre className="whitespace-pre-wrap wrap-anywhere rounded-md bg-muted p-3 font-mono text-xs">
              {JSON.stringify(response, null, 2)}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}
