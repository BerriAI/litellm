import React from "react";
import { CircleCheck, CircleX, LoaderCircle } from "lucide-react";

import { testModelGroupConnection, ModelGroupConnectionResult } from "../networking";
import { AutoRouterTestTarget } from "./build_auto_router_test_targets";

interface AutoRouterConnectionTestProps {
  accessToken: string;
  targets: AutoRouterTestTarget[];
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
  onTestComplete,
}) => {
  const [results, setResults] = React.useState<TargetResult[]>(() => targets.map(() => ({ status: "pending" })));

  React.useEffect(() => {
    let cancelled = false;
    const run = async () => {
      await Promise.all(
        targets.map(async (target, index) => {
          const result = await testModelGroupConnection(accessToken, target.modelGroup, target.mode);
          if (cancelled) return;
          const cleaned: TargetResult =
            result.status === "error" ? { status: "error", error: cleanErrorMessage(result.error) } : result;
          setResults((prev) => prev.map((r, i) => (i === index ? cleaned : r)));
        }),
      );
      if (!cancelled && onTestComplete) onTestComplete();
    };
    run();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- probes run once per mount; the parent remounts via `key` to start a fresh test, and re-running on prop identity changes would refire paid requests
  }, []);

  if (targets.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No complexity tiers are configured yet, so there is nothing to test.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <p className="mb-2 text-sm text-muted-foreground">
        Each configured tier routes to a saved model group. Test Connection sends a minimal request through the proxy to
        each one, exactly as the auto router would.
      </p>
      {targets.map((target, index) => {
        const result = results[index] ?? { status: "pending" };
        return (
          <div
            key={`${target.modelGroup}-${target.mode}`}
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
