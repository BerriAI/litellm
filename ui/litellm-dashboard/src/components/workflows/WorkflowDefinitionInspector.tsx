"use client";

import { useMemo, type ReactNode } from "react";
import { Braces, ChevronDown, Code2, MousePointer2, Sparkles, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  executionStates,
  nodeTitle,
  readWorkflowDefinition,
  type CompiledWorkflowBundle,
  type CompiledWorkflowNode,
  type WorkflowBinding,
} from "./definition";
import { WorkflowNodeStatus, type WorkflowGraphProps } from "./WorkflowGraph";

function InspectorSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3 border-t border-border pt-5">
      <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{title}</h3>
      {children}
    </section>
  );
}

function InspectorDisclosure({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Collapsible className="min-w-0 rounded-lg border border-border bg-muted/30">
      <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-xs font-medium focus-visible:outline-ring">
        {title}
        <ChevronDown aria-hidden className="size-3.5 text-muted-foreground group-data-panel-open:rotate-180" />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <pre className="max-h-64 overflow-auto border-t border-border p-3 text-xs leading-relaxed">{children}</pre>
      </CollapsibleContent>
    </Collapsible>
  );
}

function JsonDetails({ title, value }: { title: string; value: unknown }) {
  return <InspectorDisclosure title={title}>{JSON.stringify(value, null, 2)}</InspectorDisclosure>;
}

function BindingValue({ binding }: { binding: WorkflowBinding }) {
  const { t } = useTranslation();
  if ("from" in binding) {
    return (
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="rounded-md border border-border bg-muted px-2 py-1 font-mono">{binding.from}</span>
        <span className="text-muted-foreground">/</span>
        <code>{binding.output}</code>
      </div>
    );
  }
  return <JsonDetails title={t("workflowGraph.literal")} value={binding.value} />;
}

function NodeBindings({ node }: { node: CompiledWorkflowNode }) {
  const bindings = node.type === "python" ? node.inputs : { messages: node.messages };
  return (
    <dl className="space-y-3">
      {Object.entries(bindings).map(([name, value]) => (
        <div key={name} className="space-y-2">
          <dt className="break-all font-mono text-xs">{name}</dt>
          <dd>
            <BindingValue binding={value} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

function NodeSchemas({ bundle, node }: { bundle: CompiledWorkflowBundle; node: CompiledWorkflowNode }) {
  const { t } = useTranslation();
  if (node.type === "python") {
    return (
      <>
        <JsonDetails title={t("workflowGraph.inputSchema")} value={node.program.input_schema} />
        <JsonDetails title={t("workflowGraph.outputSchema")} value={node.program.output_schema} />
      </>
    );
  }
  return (
    <>
      <JsonDetails title={t("workflowGraph.inputSchema")} value={bundle.input_schema} />
      {node.response_format ? (
        <JsonDetails title={t("workflowGraph.outputSchema")} value={node.response_format.json_schema} />
      ) : (
        <p className="text-xs text-muted-foreground">{t("workflowGraph.textOutput")}</p>
      )}
    </>
  );
}

function NodeConfiguration({ node }: { node: CompiledWorkflowNode }) {
  const { t } = useTranslation();
  const parameters =
    node.type === "chat_completion"
      ? Object.fromEntries(
          ["temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop"]
            .filter((key) => node[key] !== undefined)
            .map((key) => [key, node[key]]),
        )
      : {};
  return (
    <>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-3 text-xs">
        <dt className="text-muted-foreground">{t("workflowGraph.timeout")}</dt>
        <dd className="text-right tabular-nums">{t("workflowGraph.seconds", { count: node.timeout_seconds })}</dd>
        <dt className="text-muted-foreground">{t("workflowGraph.attempts")}</dt>
        <dd className="text-right tabular-nums">{node.retry.max_attempts}</dd>
        <dt className="text-muted-foreground">{t("workflowGraph.backoff")}</dt>
        <dd className="text-right tabular-nums">{t("workflowGraph.seconds", { count: node.retry.backoff_seconds })}</dd>
      </dl>
      {Object.keys(parameters).length > 0 && (
        <JsonDetails title={t("workflowGraph.modelParameters")} value={parameters} />
      )}
    </>
  );
}

export type WorkflowDefinitionInspectorProps = Pick<
  WorkflowGraphProps,
  "bundle" | "definitionKey" | "selectedNodeId" | "onNodeSelect" | "execution" | "loading" | "error"
>;

export function WorkflowDefinitionInspector({
  bundle: input,
  definitionKey,
  selectedNodeId,
  onNodeSelect,
  execution,
  loading,
  error,
}: WorkflowDefinitionInspectorProps) {
  const { t } = useTranslation();
  const definition = useMemo(() => readWorkflowDefinition(input), [input]);
  if (loading || error || definition.kind !== "ready") return null;
  const { bundle } = definition;
  const node = bundle.nodes.find((candidate) => candidate.id === selectedNodeId);
  const states = executionStates(definitionKey, execution);
  const state = node && states?.[node.id];
  const Icon = node?.type === "python" ? Code2 : Sparkles;

  return (
    <Card
      role="complementary"
      aria-label={t("workflowGraph.inspector")}
      className="min-w-0 gap-5 p-5 lg:max-h-[640px] lg:overflow-y-auto"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">{t("workflowGraph.definition")}</h2>
        {node && (
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label={t("workflowGraph.clearSelection")}
            onClick={() => onNodeSelect(null)}
          >
            <X />
          </Button>
        )}
      </div>
      {!node ? (
        <>
          <div className="space-y-2">
            <Braces className="size-5 text-info" aria-hidden />
            <h3 className="break-all text-lg font-medium">{bundle.name}</h3>
            <p className="text-xs text-muted-foreground">
              {t("workflowGraph.workflowTimeout", { seconds: bundle.timeout_seconds })}
            </p>
          </div>
          <div className="flex gap-2 rounded-lg bg-muted p-3 text-xs text-muted-foreground">
            <MousePointer2 className="size-4 shrink-0" aria-hidden />
            {t("workflowGraph.selectHint")}
          </div>
          <InspectorSection title={t("workflowGraph.schemas")}>
            <JsonDetails title={t("workflowGraph.inputSchema")} value={bundle.input_schema} />
          </InspectorSection>
          <InspectorSection title={t("workflowGraph.output")}>
            <BindingValue binding={bundle.output} />
          </InspectorSection>
        </>
      ) : (
        <>
          <div className="space-y-3">
            <Badge variant="secondary" className="gap-2">
              <Icon className="size-3.5 text-info" aria-hidden />
              {t(`workflowGraph.types.${node.type}`)}
            </Badge>
            <h3 className="break-all text-lg font-medium">{nodeTitle(node)}</h3>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <code className="break-all text-xs text-muted-foreground">{node.id}</code>
              {state && <WorkflowNodeStatus state={state} />}
            </div>
          </div>
          <InspectorSection title={t("workflowGraph.bindings")}>
            <NodeBindings node={node} />
          </InspectorSection>
          <InspectorSection title={t("workflowGraph.schemas")}>
            <NodeSchemas bundle={bundle} node={node} />
          </InspectorSection>
          <InspectorSection title={t("workflowGraph.configuration")}>
            <NodeConfiguration node={node} />
          </InspectorSection>
          {node.type === "python" && (
            <InspectorSection title={t("workflowGraph.program")}>
              <InspectorDisclosure title={t("workflowGraph.source")}>{node.program.source}</InspectorDisclosure>
              <p className="break-all font-mono text-xs text-muted-foreground">SHA-256 {node.program.source_sha256}</p>
            </InspectorSection>
          )}
        </>
      )}
    </Card>
  );
}
