"use client";

import * as React from "react";

import { Item, ItemContent, ItemGroup, ItemMedia } from "@/components/ui/item";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";

import { NotListed } from "./HubEmpty";
import { InSheet } from "./InspectorHeader";
import { useOpenHref } from "./useHubUrlState";
import { useIsNarrow } from "./useIsNarrow";

function ListSkeleton() {
  return (
    <ItemGroup className="gap-0.5 p-3" aria-busy="true">
      {[0, 1, 2].map((i) => (
        <div key={i} role="listitem">
          <Item size="sm">
            <ItemMedia variant="icon" className="size-5">
              <Skeleton className="size-5 rounded-full" />
            </ItemMedia>
            <ItemContent className="gap-1.5">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-48" />
            </ItemContent>
          </Item>
        </div>
      ))}
    </ItemGroup>
  );
}

function DetailSkeleton() {
  return (
    <div className="flex flex-col gap-6 p-6" aria-busy="true">
      <Skeleton className="h-5 w-48" />
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="flex flex-col gap-1.5">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-4 w-16" />
          </div>
        ))}
      </div>
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

const isPlainClick = (event: React.MouseEvent) => {
  const opensElsewhere = event.metaKey || event.ctrlKey || event.shiftKey;
  return event.button === 0 && !opensElsewhere;
};

function Rows<T>({
  rows,
  activeKey,
  explicit,
  getKey,
  onSelect,
  renderItem,
}: {
  rows: T[];
  activeKey: string | undefined;
  explicit: boolean;
  getKey: (row: T) => string;
  onSelect: (key: string) => void;
  renderItem: (row: T) => React.ReactNode;
}) {
  const hrefFor = useOpenHref();
  const follow = (key: string) => (event: React.MouseEvent) => {
    if (!isPlainClick(event)) return;
    event.preventDefault();
    onSelect(key);
  };
  return (
    <ItemGroup className="gap-0.5 p-3">
      {rows.map((row) => {
        const key = getKey(row);
        const active = key === activeKey;
        return (
          <div key={key} role="listitem">
            <Item
              size="sm"
              render={
                <a href={hrefFor(key)} aria-current={active && explicit ? "true" : undefined} onClick={follow(key)} />
              }
              className={cn(active && (explicit ? "bg-muted" : "lg:bg-muted"))}
            >
              {renderItem(row)}
            </Item>
          </div>
        );
      })}
    </ItemGroup>
  );
}

const resolveSelection = <T,>(
  rows: T[],
  selectedKey: string | null,
  opened: T | undefined,
  getKey: (row: T) => string,
) => {
  const chosen = selectedKey === null ? undefined : opened ?? rows.find((row) => getKey(row) === selectedKey);
  return { chosen, selected: selectedKey === null ? rows[0] : chosen };
};

function useSheet<T>(
  chosen: T | undefined,
  missingKey: string | undefined,
  getKey: (row: T) => string,
  renderDetail: (row: T) => React.ReactNode,
) {
  const [content, setContent] = React.useState<{ row?: T; missingKey?: string }>({ row: chosen });
  if (chosen !== undefined && chosen !== content.row) setContent({ row: chosen });
  if (missingKey !== undefined && missingKey !== content.missingKey) setContent({ missingKey });
  const { row } = content;
  return {
    open: chosen !== undefined || missingKey !== undefined,
    title: row ? getKey(row) : content.missingKey ?? "Details",
    body: row ? renderDetail(row) : content.missingKey && <NotListed name={content.missingKey} />,
  };
}

export function HubMasterDetail<T>({
  switcher,
  toolbar,
  rows,
  loading,
  selectedKey,
  opened,
  openPending = false,
  getKey,
  onSelect,
  renderItem,
  renderDetail,
  empty,
  footer,
}: {
  switcher?: React.ReactNode;
  toolbar: React.ReactNode;
  rows: T[];
  loading: boolean;
  selectedKey: string | null;
  opened?: T;
  openPending?: boolean;
  getKey: (row: T) => string;
  onSelect: (key: string | null) => void;
  renderItem: (row: T) => React.ReactNode;
  renderDetail: (row: T) => React.ReactNode;
  empty: React.ReactNode;
  footer?: React.ReactNode;
}) {
  const narrow = useIsNarrow();
  const { chosen, selected } = resolveSelection(rows, selectedKey, opened, getKey);
  const isEmpty = !loading && rows.length === 0;
  const settled = !loading && !openPending && !isEmpty;
  const missingKey = selectedKey !== null && chosen === undefined && settled ? selectedKey : undefined;
  const sheet = useSheet(chosen, missingKey, getKey, renderDetail);
  const listOnly = isEmpty && chosen === undefined && !openPending;
  const detail = () => {
    if (loading || (selected === undefined && openPending)) return <DetailSkeleton />;
    if (selected === undefined) return selectedKey && <NotListed name={selectedKey} />;
    return renderDetail(selected);
  };
  const list = isEmpty ? (
    empty
  ) : (
    <>
      <Rows
        rows={rows}
        activeKey={selected && getKey(selected)}
        explicit={chosen !== undefined}
        getKey={getKey}
        onSelect={onSelect}
        renderItem={renderItem}
      />
      {footer}
    </>
  );
  return (
    <div
      className={cn(
        "grid min-h-0 grid-cols-1 overflow-hidden rounded-xl border border-border lg:h-[calc(100dvh-18.875rem)] lg:min-h-[560px]",
        !listOnly && "lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]",
      )}
    >
      <div className={cn("flex min-h-0 flex-col", !listOnly && "lg:border-r lg:border-border")}>
        <div className="flex flex-col gap-3 border-b border-border p-3">
          {switcher}
          <div className="flex flex-col gap-2">{toolbar}</div>
        </div>
        <ScrollArea className="min-h-0 flex-1">{loading ? <ListSkeleton /> : list}</ScrollArea>
      </div>
      {!listOnly && <ScrollArea className="hidden min-h-0 lg:block">{detail()}</ScrollArea>}
      <DetailSheet open={narrow && sheet.open} title={sheet.title} onClose={() => onSelect(null)}>
        {sheet.body}
      </DetailSheet>
    </div>
  );
}

function DetailSheet({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const body = React.useRef<HTMLDivElement>(null);
  return (
    <Sheet open={open} onOpenChange={(next) => !next && onClose()}>
      <SheetContent
        side="bottom"
        showCloseButton={false}
        initialFocus={body}
        className="max-h-[85dvh] gap-0 rounded-t-xl p-0 lg:hidden"
      >
        <SheetTitle className="sr-only">{title}</SheetTitle>
        <div ref={body} tabIndex={-1} className="min-h-0 overflow-y-auto outline-none">
          <InSheet.Provider value>{children}</InSheet.Provider>
        </div>
      </SheetContent>
    </Sheet>
  );
}
