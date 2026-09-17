"use client";

import React, { useState } from "react";
import { CircleAlert, CircleCheck, CirclePlay } from "lucide-react";

import ModelSelector from "@/components/common_components/ModelSelector";
import { getProxyBaseUrl } from "@/components/networking";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PromptCachingCallResult, PromptCachingVerdict, runPromptCachingTest } from "./promptCachingTest";

interface PromptCachingTestCardProps {
  accessToken: string;
  enabled: boolean;
  runTest?: typeof runPromptCachingTest;
}

interface TestOutcome {
  first: PromptCachingCallResult;
  second: PromptCachingCallResult;
  verdict: PromptCachingVerdict;
}

const formatCost = (cost: number | null) => (cost === null ? "-" : `$${cost.toFixed(6)}`);

const VERDICT_COPY: Record<
  PromptCachingVerdict,
  { variant: "success" | "info" | "error"; text: (o: TestOutcome) => string }
> = {
  injected: {
    variant: "success",
    text: (o) =>
      `LiteLLM injected cache_control: call 1 wrote ${o.first.cacheCreationTokens} tokens to the cache and call 2 read ${o.second.cacheReadTokens} tokens.`,
  },
  cache_hit_only: {
    variant: "info",
    text: () => "Cache read on both calls: cache_control was on the wire and a warm cache already existed.",
  },
  not_injected: {
    variant: "error",
    text: () =>
      "No cache activity. LiteLLM did not inject cache_control for this model. Check the toggle above, that the model is an Anthropic or Bedrock Claude model with prompt caching support, and that the key does not disable it.",
  },
};

const CallRow = ({ label, call }: { label: string; call: PromptCachingCallResult }) => (
  <TableRow>
    <TableCell className="font-medium">{label}</TableCell>
    <TableCell>{call.promptTokens}</TableCell>
    <TableCell>{call.cacheCreationTokens}</TableCell>
    <TableCell>{call.cacheReadTokens}</TableCell>
    <TableCell>{formatCost(call.responseCost)}</TableCell>
    <TableCell>{call.durationMs} ms</TableCell>
  </TableRow>
);

const PromptCachingTestCard: React.FC<PromptCachingTestCardProps> = ({
  accessToken,
  enabled,
  runTest: runTestImpl = runPromptCachingTest,
}) => {
  const [model, setModel] = useState<string | null>(null);
  const [runningCall, setRunningCall] = useState<0 | 1 | 2>(0);
  const [outcome, setOutcome] = useState<TestOutcome | null>(null);
  const [error, setError] = useState<string | null>(null);

  const running = runningCall !== 0;

  const handleRunTest = async () => {
    if (!model || running) {
      return;
    }
    setRunningCall(1);
    setOutcome(null);
    setError(null);
    try {
      const testOptions = { accessToken, model, baseUrl: getProxyBaseUrl(), onCallStart: setRunningCall };
      setOutcome(await runTestImpl(testOptions));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunningCall(0);
    }
  };

  const verdict = outcome ? VERDICT_COPY[outcome.verdict] : null;

  return (
    <Card>
      <CardContent>
        <CardTitle>Test prompt caching</CardTitle>
        <p className="mt-1 break-words text-xs text-muted-foreground">
          Sends two identical requests with a ~9k token system prompt and no cache_control of its own. If LiteLLM
          injects the breakpoints, the first call writes the cache and the second reads it.
        </p>

        {!enabled && (
          <Alert variant="info" className="mt-4">
            <CircleAlert />
            <AlertTitle>Automatic prompt caching is off</AlertTitle>
            <AlertDescription>
              Turn it on above to test injection, or run anyway to confirm nothing is injected.
            </AlertDescription>
          </Alert>
        )}

        <div className="mt-4 flex flex-wrap items-end gap-4">
          <div className="min-w-64">
            <ModelSelector
              accessToken={accessToken}
              value={model}
              onChange={setModel}
              disabled={running}
              labelText="Model"
            />
          </div>
          <Button onClick={handleRunTest} disabled={running || !model}>
            <CirclePlay />
            Run test
          </Button>
          {running && <span className="text-sm text-muted-foreground">Sending call {runningCall} of 2...</span>}
        </div>

        {error && (
          <Alert variant="error" className="mt-4">
            <CircleAlert />
            <AlertTitle>Test failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {outcome && (
          <div className="mt-4">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead />
                  <TableHead>Prompt tokens</TableHead>
                  <TableHead>Cache write</TableHead>
                  <TableHead>Cache read</TableHead>
                  <TableHead>Cost</TableHead>
                  <TableHead>Latency</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <CallRow label="Call 1" call={outcome.first} />
                <CallRow label="Call 2" call={outcome.second} />
              </TableBody>
            </Table>
            {verdict && (
              <Alert variant={verdict.variant} className="mt-4">
                {outcome.verdict === "not_injected" ? <CircleAlert /> : <CircleCheck />}
                <AlertDescription>{verdict.text(outcome)}</AlertDescription>
              </Alert>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PromptCachingTestCard;
