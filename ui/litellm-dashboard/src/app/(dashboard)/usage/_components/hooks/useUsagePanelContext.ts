import { useCallback, useLayoutEffect, useRef } from "react";

/** Keep the dashboard's nested scroll container in place across dialog focus changes. */
export function useUsagePanelContext(open: boolean) {
  const pageRef = useRef<HTMLDivElement>(null);
  const snapshot = useRef<{
    scroll: { element: HTMLElement; top: number; left: number }[];
    focus: HTMLElement | null;
  } | null>(null);
  const remember = useCallback(() => {
    const scroll: { element: HTMLElement; top: number; left: number }[] = [];
    for (let element = pageRef.current?.parentElement; element; element = element.parentElement) {
      scroll.push({ element, top: element.scrollTop, left: element.scrollLeft });
    }
    const focus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    snapshot.current = { scroll, focus };
  }, []);

  useLayoutEffect(() => {
    const saved = snapshot.current;
    if (!saved) return;
    const restore = () => {
      for (const { element, top, left } of saved.scroll) {
        element.scrollTop = top;
        element.scrollLeft = left;
      }
    };
    restore();
    const frame = requestAnimationFrame(() => {
      if (!open) saved.focus?.focus({ preventScroll: true });
      restore();
      if (!open) snapshot.current = null;
    });
    return () => cancelAnimationFrame(frame);
  }, [open]);
  return { pageRef, remember };
}
