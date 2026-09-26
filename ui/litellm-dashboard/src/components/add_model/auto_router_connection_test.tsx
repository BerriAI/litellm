import React from "react";
import { CircleCheck, CircleX, LoaderCircle } from "lucide-react";

import {
  testModelGroupConnection,
  ModelGroupConnectionResult,
  testAutoRouterRouting,
  AutoRouterRoutingTestRequest,
} from "../networking";
import { AutoRouterTestTarget } from "./build_auto_router_test_targets";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { jevClassifierConfigSchema } from "./jev_classifier_config";

interface AutoRouterConnectionTestProps {
  accessToken: string;
  targets: AutoRouterTestTarget[];
  jevRequest?: AutoRouterRoutingTestRequest;
  onTestComplete?: () => void;
}

type TargetResult = { status: "pending" } | ModelGroupConnectionResult;

const cleanErrorMessage = (error: string): string => {
  const mainError = error.split("stack trace:")[0].trim();
  return mainError.replace(/^litellm\.(.*?)Error: /, "");
};

const AutoRouterConnectionTest: React.FC<AutoRouterConnectionTestProps> = ({
  accessToken,
  targets,
  jevRequest,
  onTestComplete,
}) => {
  const [results, setResults] = React.useState<TargetResult[]>(() => targets.map(() => ({ status: "pending" })));
  const [jevResult, setJevResult] = React.useState<TargetResult>({ status: "pending" });
  const decisionConfig = jevClassifierConfigSchema.safeParse(
    jevRequest?.complexity_router_config.jev_classifier_config,
  );
  const decisionModelLabel = decisionConfig.success && decisionConfig.data.provider === "laya" ? "Laya" : "Jev";

  React.useEffect(() => {
    let cancelled = false;
    const probeJev = async () => {
      if (!jevRequest) return;
      const response = await testAutoRouterRouting(accessToken, jevRequest);
      if (cancelled) return;
      if (response.status === "error") {
        setJevResult(response);
        return;
      }
      const decision = response.result.routing_decision;
      setJevResult(
        decision.cause === "jev_classifier"
          ? { status: "success" }
          : {
              status: "error",
              error: `${decisionModelLabel} was not reached successfully (routing cause: ${decision.cause ?? "unknown"})`,
            },
      );
    };
    const run = async () => {
      await Promise.all([
        probeJev(),
        ...targets.map(async (target, index) => {
          const result = target.requestParams
            ? await testModelGroupConnection(accessToken, target.modelGroup, target.mode, target.requestParams)
            : await testModelGroupConnection(accessToken, target.modelGroup, target.mode);
          if (cancelled) return;
          const cleaned: TargetResult =
            result.status === "error" ? { status: "error", error: cleanErrorMessage(result.error) } : result;
          setResults((prev) => prev.map((r, i) => (i === index ? cleaned : r)));
        }),
      ]);
      if (!cancelled && onTestComplete) onTestComplete();
    };
    run();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- probes run once per mount; the parent remounts via `key` to start a fresh test, and re-running on prop identity changes would refire paid requests
  }, []);

  if (targets.length === 0 && !jevRequest) {
    return (
      <p className="text-sm text-muted-foreground">
        No complexity tiers are configured yet, so there is nothing to test.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <p className="mb-2 text-sm text-muted-foreground">
        Test Connection sends a minimal request to every configured tier, classifier, default, and embedding model. The
        classifier probe includes its reasoning effort override.
      </p>
      {jevRequest && (
        <div role="status" aria-label={`${decisionModelLabel} connection`} className="rounded-lg border p-3 text-sm">
          <strong>{decisionModelLabel} Classifier</strong>
          <p>
            {jevResult.status === "pending" && `Testing ${decisionModelLabel} classification`}
            {jevResult.status === "success" && `${decisionModelLabel} classification succeeded`}
            {jevResult.status === "error" && jevResult.error}
          </p>
        </div>
      )}
      {targets.map((target, index) => {
        const result = results[index] ?? { status: "pending" };
        return (
          <div
            key={`${target.labels.join("-")}-${target.modelGroup}-${target.mode}`}
            data-testid="auto-router-test-row"
            className="flex items-start gap-3 rounded-lg border p-3"
          >
            <div className="pt-0.5">
              {result.status === "pending" && (
                <LoaderCircle className="size-5 animate-spin text-muted-foreground" data-testid="test-status-pending" />
              )}
              {result.status === "success" && (
                <CircleCheck className="size-5 text-primary" data-testid="test-status-success" />
              )}
              {result.status === "error" && (
                <CircleX className="size-5 text-destructive" data-testid="test-status-error" />
              )}
            </div>
            <div className="min-w-0 flex-1 text-sm">
              <span className="font-medium">{target.labels.join(", ")}</span>{" "}
              <span className="text-muted-foreground">
                {"->"} {target.modelGroup}
                {target.mode === "embedding" ? " (embedding)" : ""}
              </span>
              {result.status === "error" && (
                <p className="mt-1 text-xs text-destructive" data-testid="test-error-message">
                  {result.error}
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default AutoRouterConnectionTest;

export function AutoRouterConnectionTestDialog({
  open,
  onClose,
  testId,
  ...props
}: AutoRouterConnectionTestProps & { open: boolean; onClose: () => void; testId: number }) {
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>Connection Test Results</DialogTitle>
        </DialogHeader>
        {open && <AutoRouterConnectionTest key={testId} {...props} />}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
