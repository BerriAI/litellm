"use client";

import { type RefObject, useRef, useState } from "react";
import { useDefaultLayout } from "react-resizable-panels";
import { useResizeObserver } from "usehooks-ts";

import { useShortcut } from "@/components/shared/useShortcut";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/Resizable";
import { TabsContent } from "@/components/ui/tabs";

import type { RunSelection } from "../../routing";
import type { Trace } from "../../types";
import { TraceConversation, type ConversationTracePaging } from "../conversation/TraceConversation";
import { DetailPane } from "../span/DetailPane";
import { SpanTree } from "../tree/SpanTree";
import type { TreeLayout } from "../tree/TreeRows";
import { useRunTree } from "./useRunTree";

const SPLIT_LAYOUT_ID = "litellm.lens.runSplit";
const SIDE_BY_SIDE_MIN_PX = 640;

interface RunBodyProps {
  trace: Trace;
  accessToken: string;
  selection: RunSelection;
  embedded: boolean;
  stale: boolean;
  conversationPaging: ConversationTracePaging;
}

/** Tree + detail pane for one loaded run. Arrows move and fold steps; J/K also move unless the drawer owns them. */
export function RunBody({ trace, accessToken, selection, embedded, stale, conversationPaging }: RunBodyProps) {
  const { view, setView, stepQuery, setStepQuery, errorsOnly, setErrorsOnly } = selection;
  const tree = useRunTree(trace, selection);
  const [detailOpen, setDetailOpen] = useState(true);
  const [layout, setLayout] = useState<TreeLayout>("tree");
  const splitRef = useRef<HTMLDivElement>(null);
  const { width = SIDE_BY_SIDE_MIN_PX } = useResizeObserver({ ref: splitRef as RefObject<HTMLDivElement> });
  const orientation = width >= SIDE_BY_SIDE_MIN_PX ? "horizontal" : "vertical";
  const saved = useDefaultLayout({ id: `${SPLIT_LAYOUT_ID}.${orientation}`, panelIds: ["steps", "detail"] });
  const select = (id: string) => {
    tree.select(id);
    setDetailOpen(true);
  };

  const active = !stale && view === "steps";
  const pane = { layer: "pane", enabled: active } as const;
  useShortcut("down", () => tree.moveBy(1), { ...pane, description: "step" });
  useShortcut("up", () => tree.moveBy(-1), { ...pane, description: "step" });
  useShortcut("j", () => tree.moveBy(1), { ...pane, enabled: active && !embedded, description: "move" });
  useShortcut("k", () => tree.moveBy(-1), { ...pane, enabled: active && !embedded, description: "move" });
  useShortcut("left", () => tree.fold(false), { ...pane, description: "fold" });
  useShortcut("right", () => tree.fold(true), { ...pane, description: "fold" });
  useShortcut("escape", () => setDetailOpen(false), { ...pane, enabled: active && detailOpen, description: "close" });

  if (view === "conversation")
    return (
      <TabsContent value="conversation" className="flex min-h-0 flex-1">
        <TraceConversation
          trace={trace}
          accessToken={accessToken}
          paging={conversationPaging}
          onOpenStep={(id) => {
            select(id);
            setView("steps");
          }}
        />
      </TabsContent>
    );

  return (
    <TabsContent value="steps" ref={splitRef} className="flex min-h-0 flex-1">
      <ResizablePanelGroup
        key={orientation}
        orientation={orientation}
        defaultLayout={detailOpen ? saved.defaultLayout : undefined}
        onLayoutChanged={detailOpen ? saved.onLayoutChanged : undefined}
      >
        <ResizablePanel id="steps" defaultSize="40%" minSize={orientation === "horizontal" ? 300 : 160} maxSize="65%">
          <SpanTree
            rows={tree.rows}
            summary={trace.summary}
            selectedId={tree.selectedId}
            layout={layout}
            onLayoutChange={setLayout}
            hideFramework={tree.hideFramework}
            onSelect={select}
            onToggleHideFramework={tree.setHideFramework}
            onToggleSpan={tree.toggleSpan}
            onToggleGroup={tree.toggleGroup}
            onLoadMore={tree.loadMore}
            onOpenDetails={detailOpen ? undefined : () => setDetailOpen(true)}
            query={stepQuery}
            onQueryChange={setStepQuery}
            errorsOnly={errorsOnly}
            onErrorsOnlyChange={setErrorsOnly}
            filtering={tree.filtering}
            onClearFilters={() => {
              setStepQuery("");
              setErrorsOnly(false);
            }}
            onCollapseAll={tree.collapseAll}
          />
        </ResizablePanel>
        {detailOpen && (
          <>
            <ResizableHandle withHandle className="bg-border hover:bg-trace-chain" />
            <ResizablePanel id="detail" minSize={orientation === "horizontal" ? 360 : 200} className="min-w-0">
              <DetailPane
                trace={trace}
                row={tree.selectedRow}
                accessToken={accessToken}
                spanTab={selection.spanTab}
                onSpanTabChange={selection.setSpanTab}
                onClose={() => setDetailOpen(false)}
              />
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
    </TabsContent>
  );
}
