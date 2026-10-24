import type { VirtualItem, Virtualizer } from "@tanstack/react-virtual";
import { useEffect, useLayoutEffect, useRef } from "react";

import { LogEntry } from "../types";

interface ScrollAnchor {
  readonly requestId: string;
  /** The row's top relative to the top of the scroller's viewport. */
  readonly viewportOffset: number;
}

interface AnchorInputs {
  readonly logs: LogEntry[];
  readonly selectedRequestId: string | undefined;
}

type SessionListVirtualizer = Virtualizer<HTMLDivElement, Element>;

const indexOfRequest = (logs: LogEntry[], requestId: string | undefined) =>
  requestId ? logs.findIndex((row) => row.request_id === requestId) : -1;

// Virtualizer positions count from the top of the list; the scroller can show
// more content above it. offsetTop needs the scroller to be the list's offsetParent.
const listOffsetIn = (list: HTMLElement | null) => list?.offsetTop ?? 0;

// measurementsCache lags behind rows measured as a commit attaches them until the
// virtualizer recomputes it; getTotalSize() makes it do so.
const currentMeasurements = (virtualizer: SessionListVirtualizer) => {
  virtualizer.getTotalSize();
  return virtualizer.measurementsCache;
};

// The row to hold in place: the selected row while it is on screen, otherwise
// the first row on screen once the user has scrolled. With neither (top of the
// list, selected row off screen) nothing is held, so longer calls that arrive
// later show up first.
function findAnchor(
  virtualizer: SessionListVirtualizer,
  { logs, selectedRequestId }: AnchorInputs,
  scrollTop: number,
  listOffset: number,
): ScrollAnchor | null {
  const viewportStart = scrollTop - listOffset;
  const viewportEnd = viewportStart + (virtualizer.scrollRect?.height ?? 0);
  const selected = currentMeasurements(virtualizer)[indexOfRequest(logs, selectedRequestId)];
  let item: VirtualItem | undefined;
  if (selected !== undefined && selected.end > viewportStart && selected.start < viewportEnd) item = selected;
  else if (scrollTop > 0) item = virtualizer.getVirtualItemForOffset(Math.max(0, viewportStart));
  if (!item || !logs[item.index]) return null;
  return { requestId: logs[item.index].request_id, viewportOffset: item.start - viewportStart };
}

interface SessionScrollAnchorOptions {
  readonly scroller: HTMLDivElement | null;
  /** The element the virtualizer's rows are laid out in, inside the scroller. */
  readonly list: HTMLElement | null;
  readonly logs: LogEntry[];
  readonly selectedRequestId: string | undefined;
}

/**
 * Keeps the session sidebar steady while pages arrive and the list re-sorts:
 * the row the user is looking at stays where it is on screen, and the selected
 * row is scrolled into view when it first appears or the selection changes.
 */
export function useSessionScrollAnchor(
  virtualizer: SessionListVirtualizer,
  { scroller, list, logs, selectedRequestId }: SessionScrollAnchorOptions,
) {
  const anchorRef = useRef<ScrollAnchor | null>(null);
  const revealedRequestIdRef = useRef<string | undefined>(undefined);
  // scrollTop after this hook's last scroll; a scroll event that ends elsewhere is the user's.
  const ownScrollTopRef = useRef<number | null>(null);
  const inputsRef = useRef<AnchorInputs>({ logs, selectedRequestId });
  // Rows measured after the anchor was restored (they scrolled into the render
  // window) move the rows below them; restoring again keeps the held row put.
  const totalSize = virtualizer.getTotalSize();

  useLayoutEffect(() => {
    const inputs = { logs, selectedRequestId };
    inputsRef.current = inputs;
    const viewportHeight = virtualizer.scrollRect?.height ?? 0;
    if (!scroller || viewportHeight === 0) return;
    const listOffset = listOffsetIn(list);
    const measurements = currentMeasurements(virtualizer);
    // Returns whether the scroll got there: the browser stops it at the end of the list.
    const scrollTo = (target: number) => {
      const top = Math.max(0, target);
      if (top === scroller.scrollTop) return true;
      scroller.scrollTo({ top });
      ownScrollTopRef.current = scroller.scrollTop;
      // A browser reports the scroll after the next paint, so the virtualizer would paint
      // that frame with the rows of the old position: a blank sidebar each time a page lands.
      scroller.dispatchEvent(new Event("scroll"));
      return Math.abs(scroller.scrollTop - top) < 1;
    };

    const anchor = anchorRef.current;
    const anchored = anchor ? measurements[indexOfRequest(logs, anchor.requestId)] : undefined;
    let restoredInFull = true;
    if (anchor && anchored) {
      // A selected row held partly off screen is pulled fully into view: it is revealed
      // at its estimated height, and its measured height can push it past the fold.
      const viewportOffset =
        anchor.requestId === selectedRequestId
          ? Math.max(0, Math.min(anchor.viewportOffset, viewportHeight - anchored.size))
          : anchor.viewportOffset;
      restoredInFull = scrollTo(listOffset + anchored.start - viewportOffset);
    }

    const selected = measurements[indexOfRequest(logs, selectedRequestId)];
    if (selected && revealedRequestIdRef.current !== selectedRequestId) {
      const onScreen = () =>
        listOffset + selected.start >= scroller.scrollTop &&
        listOffset + selected.end <= scroller.scrollTop + viewportHeight;
      // Centered: rows above it are measured only after this scroll, and their real heights move it.
      if (!onScreen()) scrollTo(listOffset + selected.start - (viewportHeight - selected.size) / 2);
      // The list may not reach that far yet (its end, before the rows there are measured):
      // try again when sizes change, until the user scrolls.
      if (onScreen()) revealedRequestIdRef.current = selectedRequestId;
    }

    // A restore cut short by a list whose rendered height does not yet count the rows measured
    // in this commit keeps its anchor: the render for the new height restores it again.
    const heightPending = (list?.offsetHeight ?? 0) < virtualizer.getTotalSize();
    if (restoredInFull || !heightPending) {
      anchorRef.current = findAnchor(virtualizer, inputs, scroller.scrollTop, listOffset);
    }
  }, [virtualizer, scroller, list, logs, selectedRequestId, totalSize]);

  useEffect(() => {
    if (!scroller) return;
    const onScroll = () => {
      // The anchor was set after this hook's own scroll; rows measured since then would
      // make it pick another row before the next restore.
      if (scroller.scrollTop === ownScrollTopRef.current) return;
      revealedRequestIdRef.current = inputsRef.current.selectedRequestId;
      anchorRef.current = findAnchor(virtualizer, inputsRef.current, scroller.scrollTop, listOffsetIn(list));
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => scroller.removeEventListener("scroll", onScroll);
  }, [virtualizer, scroller, list]);
}
