"use client";

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { fitViewport, MIN_SCALE, NODE_HEIGHT, NODE_WIDTH, zoomViewport, type Size, type Viewport } from "./viewport";

export function useWorkflowViewport(graph: Size) {
  const surfaceRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; view: Viewport } | null>(null);
  const [size, setSize] = useState<Size>({ width: 600, height: 640 });
  const [manualView, setManualView] = useState<Viewport | null>(null);
  const { width, height } = graph;
  const fitted = useMemo(() => fitViewport(size, { width, height }), [size, width, height]);
  const minimumScale = Math.min(MIN_SCALE, fitted.scale);
  const view = manualView ?? fitted;

  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
        setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    observer.observe(surface);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const bounds = surface.getBoundingClientRect();
      const delta = event.deltaMode === 1 ? 16 : 1;
      setManualView((current) => {
        const previous = current ?? fitted;
        if (event.ctrlKey || event.metaKey) {
          return zoomViewport(
            previous,
            Math.exp(-event.deltaY * delta * 0.005),
            { x: event.clientX - bounds.left, y: event.clientY - bounds.top },
            minimumScale,
          );
        }
        return { ...previous, x: previous.x - event.deltaX * delta, y: previous.y - event.deltaY * delta };
      });
    };
    surface.addEventListener("wheel", handleWheel, { passive: false });
    return () => surface.removeEventListener("wheel", handleWheel);
  }, [fitted, minimumScale]);

  function zoom(factor: number) {
    setManualView(zoomViewport(view, factor, { x: size.width / 2, y: size.height / 2 }, minimumScale));
  }

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || (event.target instanceof Element && event.target.closest("button"))) return;
    drag.current = { x: event.clientX, y: event.clientY, view };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus();
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!drag.current) return;
    const start = drag.current;
    setManualView({
      ...start.view,
      x: start.view.x + event.clientX - start.x,
      y: start.view.y + event.clientY - start.y,
    });
  }

  function endDrag() {
    drag.current = null;
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.target !== event.currentTarget) return;
    const movements: Record<string, [number, number]> = {
      ArrowLeft: [48, 0],
      ArrowRight: [-48, 0],
      ArrowUp: [0, 48],
      ArrowDown: [0, -48],
    };
    const movement = movements[event.key];
    if (movement) {
      event.preventDefault();
      setManualView({ ...view, x: view.x + movement[0], y: view.y + movement[1] });
    }
    if (["+", "=", "-", "0"].includes(event.key)) {
      event.preventDefault();
      if (event.key === "0") setManualView(null);
      else zoom(event.key === "-" ? 1 / 1.2 : 1.2);
    }
  }

  function revealNode(position: { x: number; y: number }) {
    const left = view.x + position.x * view.scale;
    const top = view.y + position.y * view.scale;
    const outsideWidth = left < 0 || left + NODE_WIDTH * view.scale > size.width;
    const outsideHeight = top < 0 || top + NODE_HEIGHT * view.scale > size.height;
    if (outsideWidth || outsideHeight) {
      setManualView({
        ...view,
        x: size.width / 2 - (position.x + NODE_WIDTH / 2) * view.scale,
        y: size.height / 2 - (position.y + NODE_HEIGHT / 2) * view.scale,
      });
    }
  }

  return {
    surfaceRef,
    view,
    canZoomOut: view.scale > minimumScale,
    zoom,
    fit: () => setManualView(null),
    revealNode,
    handlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
      onLostPointerCapture: endDrag,
      onKeyDown,
    },
  };
}
