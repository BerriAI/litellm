import type { Row, Table } from "@tanstack/react-table";

import { idsBetween, paintRowSelection } from "./rowSelectionRange";

const EDGE_ZONE_PX = 40;
const MAX_SCROLL_STEP_PX = 20;

export const autoScrollStep = (pointerY: number, top: number, bottom: number): number => {
  const intoTopZone = Math.min(top + EDGE_ZONE_PX - pointerY, EDGE_ZONE_PX);
  if (intoTopZone > 0) return -Math.ceil((MAX_SCROLL_STEP_PX * intoTopZone) / EDGE_ZONE_PX);
  const intoBottomZone = Math.min(pointerY - (bottom - EDGE_ZONE_PX), EDGE_ZONE_PX);
  if (intoBottomZone > 0) return Math.ceil((MAX_SCROLL_STEP_PX * intoBottomZone) / EDGE_ZONE_PX);
  return 0;
};

const scrollableAncestor = (element: HTMLElement | null): HTMLElement | null => {
  if (element === null || element === document.body || element === document.documentElement) return null;
  const { overflowY } = getComputedStyle(element);
  if ((overflowY === "auto" || overflowY === "scroll") && element.scrollHeight > element.clientHeight) return element;
  return scrollableAncestor(element.parentElement);
};

const rowIdAt = (tableElement: HTMLTableElement, x: number, y: number): string | null => {
  const rowElement = document.elementFromPoint(x, y)?.closest<HTMLElement>("tr[data-row-id]");
  if (rowElement == null || !tableElement.contains(rowElement)) return null;
  return rowElement.dataset.rowId ?? null;
};

interface DragSelection<TData> {
  table: Table<TData>;
  anchor: Row<TData>;
  tableElement: HTMLTableElement;
  pointer: { x: number; y: number };
  waitForRowChange: boolean;
  onSelectionStart?: () => void;
}

export function startDragSelection<TData>({
  table,
  anchor,
  tableElement,
  pointer,
  waitForRowChange,
  onSelectionStart,
}: DragSelection<TData>): void {
  const rows = table.getRowModel().rows;
  const orderedIds = rows.map((row) => row.id);
  const selectableIds: ReadonlySet<string> = new Set(rows.filter((row) => row.getCanSelect()).map((row) => row.id));
  const snapshot = table.getState().rowSelection;
  const selected = !anchor.getIsSelected();
  const scrollContainer = scrollableAncestor(tableElement.parentElement);
  const previousUserSelect = document.body.style.userSelect;
  const drag = { active: false, pointerY: pointer.y, frame: 0 };

  const edges = (): [number, number] => {
    if (scrollContainer === null) return [0, window.innerHeight];
    const rect = scrollContainer.getBoundingClientRect();
    return [Math.max(rect.top, 0), Math.min(rect.bottom, window.innerHeight)];
  };
  const scrollBy = (step: number) => (scrollContainer ?? window).scrollBy(0, step);

  const paintRange = (rowId: string) => {
    const range = idsBetween(orderedIds, anchor.id, rowId).filter((id) => selectableIds.has(id));
    table.setRowSelection(paintRowSelection(snapshot, range, selected));
  };
  const paintAt = (y: number) => {
    const rowId = rowIdAt(tableElement, pointer.x, y);
    if (rowId === null || (!drag.active && rowId === anchor.id)) return;
    if (!drag.active) activate();
    window.getSelection()?.removeAllRanges();
    paintRange(rowId);
  };
  const tick = () => {
    const step = autoScrollStep(drag.pointerY, ...edges());
    if (step !== 0) {
      scrollBy(step);
      paintAt(drag.pointerY);
    }
    drag.frame = requestAnimationFrame(tick);
  };
  const activate = () => {
    drag.active = true;
    onSelectionStart?.();
    document.body.style.userSelect = "none";
    drag.frame = requestAnimationFrame(tick);
  };
  const onMove = (event: PointerEvent) => {
    drag.pointerY = event.clientY;
    paintAt(event.clientY);
  };
  const listeners = new AbortController();
  const stop = () => {
    listeners.abort();
    cancelAnimationFrame(drag.frame);
    document.body.style.userSelect = previousUserSelect;
  };

  if (!waitForRowChange) {
    activate();
    paintRange(anchor.id);
  }
  document.addEventListener("pointermove", onMove, { signal: listeners.signal });
  document.addEventListener("pointerup", stop, { signal: listeners.signal });
  document.addEventListener("pointercancel", stop, { signal: listeners.signal });
}
