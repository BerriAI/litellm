import React from "react";
import { TriangleAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import RoutingDecisionCard from "@/components/view_logs/LogDetailsDrawer/RoutingDecisionCard";
import { AutoRouterRoutingTestResult, testAutoRouterRouting } from "../networking";
import { ComplexityRouterConfigPayload } from "./build_complexity_router_config";
import { buildAutoRouterRoutingTestRequest } from "./build_auto_router_routing_test_request";

interface AutoRouterRoutingTestProps {
  accessToken: string;
  config: ComplexityRouterConfigPayload;
  defaultModel: string | undefined;
  routerName: string | undefined;
  teamId: string | undefined;
}

type TestState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: AutoRouterRoutingTestResult }
  | { status: "failed"; error: string };

const AutoRouterRoutingTest: React.FC<AutoRouterRoutingTestProps> = ({
  accessToken,
  config,
  defaultModel,
  routerName,
  teamId,
}) => {
  const [prompt, setPrompt] = React.useState<string>("");
  const [state, setState] = React.useState<TestState>({ status: "idle" });

  const send = async () => {
    setState({ status: "running" });
    const params = { prompt, config, defaultModel, routerName, teamId };
    const request = buildAutoRouterRoutingTestRequest(params);
    const response = await testAutoRouterRouting(accessToken, request);
    setState(
      response.status === "success"
        ? { status: "done", result: response.result }
        : { status: "failed", error: response.error },
    );
  };

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Send a prompt through this router&apos;s classifier to see which model it would pick, and why. The prompt is
        only classified: nothing is sent to the model it routes to.
      </p>

      <Textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder="Paste a prompt an end user would send"
        rows={4}
        data-testid="auto-router-routing-test-prompt"
      />

      <div className="flex justify-end">
        <Button
          onClick={send}
          disabled={prompt.trim().length === 0 || state.status === "running"}
          data-testid="auto-router-routing-test-send"
        >
          {state.status === "running" ? "Routing..." : "Send Test Prompt"}
        </Button>
      </div>

      {state.status === "failed" && (
        <div
          className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive"
          data-testid="auto-router-routing-test-error"
        >
          <p className="font-medium">Could not route this prompt</p>
          <p>{state.error}</p>
        </div>
      )}

      {state.status === "done" && (
        <div data-testid="auto-router-routing-test-result">
          <div className="flex items-center gap-2 py-2 text-sm">
            <span className="text-muted-foreground">Routed to</span>
            <Badge variant="secondary" data-testid="auto-router-routing-test-routed-model">
              {state.result.routed_model}
            </Badge>
            {!state.result.routed_model_configured && (
              <span
                className="flex items-center gap-1 text-warning"
                data-testid="auto-router-routing-test-unconfigured"
              >
                <TriangleAlert className="size-3.5" />
                This proxy has no model group by that name
              </span>
            )}
          </div>
          <RoutingDecisionCard decision={state.result.routing_decision} />
        </div>
      )}
    </div>
  );
};

export default AutoRouterRoutingTest;
