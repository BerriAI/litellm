"use client";

import { useId, useMemo } from "react";
import {
  ArrowDownToLine,
  Braces,
  Check,
  Circle,
  Code2,
  GitBranch,
  Loader2,
  Maximize,
  Minus,
  Plus,
  Sparkles,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";
import {
  executionStates,
  nodeTitle,
  readWorkflowDefinition,
  sequentialConnections,
  type CompiledWorkflowBundle,
  type WorkflowExecutionOverlay,
  type WorkflowNodeState,
} from "./definition";
import { graphSize, MAX_SCALE, MIN_SCALE, NODE_HEIGHT, NODE_WIDTH, nodePosition } from "./viewport";
import { useWorkflowViewport } from "./useWorkflowViewport";
import styles from "./WorkflowGraph.module.css";

export interface WorkflowGraphProps {
  bundle: unknown;
  definitionKey: string;
  selectedNodeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  execution?: WorkflowExecutionOverlay;
  loading?: boolean;
  error?: string | null;
  className?: string;
}

export function WorkflowNodeStatus({ state }: { state: WorkflowNodeState }) {
  const { t } = useTranslation();
  const icons = { pending: Circle, running: Loader2, completed: Check, failed: X, cancelled: Minus };
  const Icon = icons[state];
  return (
    <span className={styles.state} data-state={state}>
      <Icon aria-hidden className={cn("size-3", state === "running" && "motion-safe:animate-spin")} />
      {t(`workflowGraph.states.${state}`)}
    </span>
  );
}

function connectionPath(start: { x: number; y: number }, end: { x: number; y: number }) {
  const middle = (start.y + end.y) / 2;
  return `M ${start.x} ${start.y} C ${start.x} ${middle}, ${end.x} ${middle}, ${end.x} ${end.y}`;
}

function GraphConnections({ bundle }: { bundle: CompiledWorkflowBundle }) {
  const marker = useId();
  const size = graphSize(bundle.nodes.length);
  const positions = new Map(bundle.nodes.map((node, index) => [node.id, nodePosition(index)]));
  const first = nodePosition(0);
  const last = nodePosition(bundle.nodes.length - 1);
  return (
    <svg
      width={size.width}
      height={size.height}
      className="pointer-events-none absolute inset-0 overflow-visible"
      aria-hidden
    >
      <defs>
        <marker
          id={marker}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10" fill="var(--info)" />
        </marker>
      </defs>
      <g fill="none" stroke="var(--info)" strokeWidth="1.5" strokeOpacity="0.65" markerEnd={`url(#${marker})`}>
        <path d={connectionPath({ x: 200, y: 44 }, { x: first.x + NODE_WIDTH / 2, y: first.y - 6 })} />
        {sequentialConnections(bundle.nodes).map(({ source, target }) => {
          const from = positions.get(source)!;
          const to = positions.get(target)!;
          return (
            <path
              key={`${source}:${target}`}
              d={connectionPath(
                { x: from.x + NODE_WIDTH / 2, y: from.y + NODE_HEIGHT },
                { x: to.x + NODE_WIDTH / 2, y: to.y - 6 },
              )}
            />
          );
        })}
        <path
          d={connectionPath({ x: last.x + NODE_WIDTH / 2, y: last.y + NODE_HEIGHT }, { x: 200, y: size.height - 38 })}
        />
      </g>
    </svg>
  );
}

function SequentialGraph({
  bundle,
  definitionKey,
  selectedNodeId,
  onNodeSelect,
  execution,
}: Omit<WorkflowGraphProps, "bundle"> & { bundle: CompiledWorkflowBundle }) {
  const { t } = useTranslation();
  const hintId = useId();
  const size = graphSize(bundle.nodes.length);
  const { surfaceRef, view, zoom, fit, revealNode, handlers } = useWorkflowViewport(size);
  const states = executionStates(definitionKey, execution);

  return (
    <>
      <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-md border border-border bg-background/95 px-3 py-2 text-xs text-muted-foreground">
        <GitBranch className="size-3.5" aria-hidden />
        {t("workflowGraph.sequential")} <span aria-hidden>·</span>{" "}
        {t("workflowGraph.stepCount", { count: bundle.nodes.length })}
      </div>
      <div
        ref={surfaceRef}
        className={styles.surface}
        role="region"
        tabIndex={0}
        aria-label={t("workflowGraph.canvas")}
        aria-describedby={hintId}
        {...handlers}
      >
        <div
          className={styles.stage}
          style={{
            width: size.width,
            height: size.height,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
          }}
        >
          <GraphConnections bundle={bundle} />
          <div className={styles.terminal} style={{ top: 12 }}>
            <Braces className="size-3.5 text-info" aria-hidden />
            {t("workflowGraph.input")}
          </div>
          {bundle.nodes.map((node, index) => {
            const position = nodePosition(index);
            const state = states?.[node.id];
            const Icon = node.type === "python" ? Code2 : Sparkles;
            return (
              <Button
                key={node.id}
                type="button"
                variant="outline"
                className={styles.node}
                style={{ left: position.x, top: position.y, width: NODE_WIDTH, height: NODE_HEIGHT }}
                aria-label={t("workflowGraph.nodeLabel", {
                  index: index + 1,
                  title: nodeTitle(node),
                  type: t(`workflowGraph.types.${node.type}`),
                })}
                aria-pressed={selectedNodeId === node.id}
                onClick={() => onNodeSelect(node.id)}
                onFocus={() => revealNode(position)}
              >
                <span className={styles.port} style={{ top: -5 }} aria-hidden />
                <span className="flex w-full items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Icon className="size-4 text-info" aria-hidden />
                    {t(`workflowGraph.types.${node.type}`)}
                  </span>
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                </span>
                <span className="w-full truncate text-sm font-medium" title={nodeTitle(node)}>
                  {nodeTitle(node)}
                </span>
                <span className="flex w-full items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs text-muted-foreground">{node.id}</span>
                  {state && <WorkflowNodeStatus state={state} />}
                </span>
                <span className={styles.port} style={{ bottom: -5 }} aria-hidden />
              </Button>
            );
          })}
          <div className={styles.terminal} style={{ top: size.height - 32 }}>
            <ArrowDownToLine className="size-3.5 text-info" aria-hidden />
            {t("workflowGraph.output")}
          </div>
        </div>
      </div>
      <div className="pointer-events-none absolute bottom-4 left-4 right-4 flex items-end justify-between gap-2">
        <p
          id={hintId}
          className="sr-only sm:not-sr-only sm:max-w-64 sm:rounded-md sm:bg-background/95 sm:p-2 sm:text-xs sm:text-muted-foreground"
        >
          {t("workflowGraph.navigationHint")}
        </p>
        <p className="rounded-md bg-background/95 p-2 text-xs text-muted-foreground sm:hidden">
          {t("workflowGraph.panHint")}
        </p>
        <div className="pointer-events-auto flex items-center gap-1 rounded-lg border border-border bg-card p-1 shadow-sm">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={t("workflowGraph.zoomOut")}
            disabled={view.scale <= MIN_SCALE}
            onClick={() => zoom(1 / 1.2)}
          >
            <Minus />
          </Button>
          <output className="w-12 text-center text-xs tabular-nums" aria-label={t("workflowGraph.zoomLevel")}>
            {Math.round(view.scale * 100)}%
          </output>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={t("workflowGraph.zoomIn")}
            disabled={view.scale >= MAX_SCALE}
            onClick={() => zoom(1.2)}
          >
            <Plus />
          </Button>
          <span className="mx-1 h-4 border-l border-border" />
          <Button variant="ghost" size="icon-sm" aria-label={t("workflowGraph.fit")} onClick={fit}>
            <Maximize />
          </Button>
        </div>
      </div>
    </>
  );
}

export function WorkflowGraph(props: WorkflowGraphProps) {
  const { t } = useTranslation();
  const definition = useMemo(() => readWorkflowDefinition(props.bundle), [props.bundle]);
  const contentClass = "flex h-full flex-col items-center justify-center gap-3 px-8 text-center";
  if (props.loading) {
    return (
      <div className={cn(styles.canvas, props.className)} role="status" aria-label={t("workflowGraph.loading")}>
        <div className={contentClass}>
          <Skeleton className="h-24 w-64" />
          <Skeleton className="h-24 w-64" />
          <p className="text-sm text-muted-foreground">{t("workflowGraph.loading")}</p>
        </div>
      </div>
    );
  }
  if (props.error || definition.kind !== "ready") {
    const kind = props.error || definition.kind === "ready" ? "error" : definition.kind;
    const description = definition.kind === "ready" ? "" : t(`workflowGraph.${definition.kind}Description`);
    return (
      <div className={cn(styles.canvas, props.className)}>
        <div className={contentClass} role={kind === "error" || kind === "invalid" ? "alert" : "status"}>
          <GitBranch className="size-8 text-muted-foreground" aria-hidden />
          <p className="font-medium">{t(`workflowGraph.${kind}Title`)}</p>
          <p className="max-w-sm text-sm text-muted-foreground">{props.error ?? description}</p>
          {definition.kind === "unsupported" && (
            <code className="max-w-full break-all text-xs">{definition.detail}</code>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className={cn(styles.canvas, props.className)}>
      <SequentialGraph key={props.definitionKey} {...props} bundle={definition.bundle} />
    </div>
  );
}
