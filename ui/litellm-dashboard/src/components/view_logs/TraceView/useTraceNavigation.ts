import { useEffect } from "react";

import { KEY_J_LOWER, KEY_J_UPPER, KEY_K_LOWER, KEY_K_UPPER } from "../LogDetailsDrawer/constants";

const isTyping = (target: EventTarget | null): boolean =>
  target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;

/** Id `delta` steps away from `current` in `order`, clamped to the ends. */
export const stepSelection = (order: readonly string[], current: string | null, delta: number): string | null => {
  if (order.length === 0) return null;
  const index = current ? order.indexOf(current) : -1;
  if (index < 0) return order[0];
  return order[Math.min(Math.max(index + delta, 0), order.length - 1)];
};

/** J / ArrowDown selects the next span, K / ArrowUp the previous one. Esc is handled by the Sheet. */
export function useTraceNavigation(active: boolean, onStep: (delta: number) => void): void {
  useEffect(() => {
    if (!active) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (isTyping(event.target)) return;
      if ([KEY_J_LOWER, KEY_J_UPPER, "ArrowDown"].includes(event.key)) {
        event.preventDefault();
        onStep(1);
      } else if ([KEY_K_LOWER, KEY_K_UPPER, "ArrowUp"].includes(event.key)) {
        event.preventDefault();
        onStep(-1);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [active, onStep]);
}
