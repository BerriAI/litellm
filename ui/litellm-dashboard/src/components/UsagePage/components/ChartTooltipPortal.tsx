import { useCallback, useRef, type MouseEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";

const CURSOR_OFFSET_PX = 14;
const EDGE_FLIP_RATIO = 0.6;

export interface CursorPosition {
  x: number;
  y: number;
}

export function useCursorPosition() {
  const positionRef = useRef<CursorPosition>({ x: 0, y: 0 });
  const handleMouseMove = useCallback((event: MouseEvent<HTMLElement>): void => {
    positionRef.current = { x: event.clientX, y: event.clientY };
  }, []);
  return { positionRef, handleMouseMove };
}

interface ChartTooltipPortalProps {
  active: boolean;
  position: CursorPosition;
  children: ReactNode;
}

export function ChartTooltipPortal({ active, position, children }: ChartTooltipPortalProps): ReactNode {
  if (!active || typeof document === "undefined") {
    return null;
  }

  const flipX = position.x > window.innerWidth * EDGE_FLIP_RATIO;
  const flipY = position.y > window.innerHeight * EDGE_FLIP_RATIO;

  return createPortal(
    <div
      data-testid="chart-tooltip-portal"
      style={{
        position: "fixed",
        left: position.x + (flipX ? -CURSOR_OFFSET_PX : CURSOR_OFFSET_PX),
        top: position.y + (flipY ? -CURSOR_OFFSET_PX : CURSOR_OFFSET_PX),
        transform: `translate(${flipX ? "-100%" : "0"}, ${flipY ? "-100%" : "0"})`,
        zIndex: 9999,
        pointerEvents: "none",
      }}
    >
      {children}
    </div>,
    document.body,
  );
}
