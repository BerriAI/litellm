"use client";

import * as React from "react";
import { Maximize2Icon, Minimize2Icon } from "lucide-react";

import { cn } from "@/lib/cva.config";
import { Button } from "@/components/ui/button";
import { SheetContent } from "@/components/ui/sheet";
import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";

type ResizableSheetContentProps = Omit<React.ComponentProps<typeof SheetContent>, "side" | "style"> & {
  storageKey: string;
  defaultWidthPercent?: number;
  minWidthPercent?: number;
};

function clampWidth(value: number, min: number) {
  return Math.min(100, Math.max(min, value));
}

function readStoredWidth(storageKey: string, defaultWidthPercent: number, minWidthPercent: number) {
  const stored = Number(getLocalStorageItem(storageKey));
  if (!Number.isFinite(stored)) return defaultWidthPercent;
  if (stored < minWidthPercent || stored > 100) return defaultWidthPercent;
  return stored;
}

function ResizableSheetContent({
  className,
  children,
  storageKey,
  defaultWidthPercent = 75,
  minWidthPercent = 40,
  ...props
}: ResizableSheetContentProps) {
  const [width, setWidth] = React.useState(() => readStoredWidth(storageKey, defaultWidthPercent, minWidthPercent));
  const widthRef = React.useRef(width);
  const lastNonFullWidthRef = React.useRef(defaultWidthPercent);
  const cleanupRef = React.useRef<(() => void) | null>(null);
  const isFull = width >= 100;

  React.useEffect(() => () => cleanupRef.current?.(), []);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const onMove = (e: PointerEvent) => {
      const next = clampWidth(((window.innerWidth - e.clientX) / window.innerWidth) * 100, minWidthPercent);
      widthRef.current = next;
      setWidth(next);
    };
    const onUp = () => {
      cleanupRef.current?.();
      cleanupRef.current = null;
      setLocalStorageItem(storageKey, String(widthRef.current));
    };
    cleanupRef.current = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const toggle = () => {
    if (isFull) {
      const next = clampWidth(lastNonFullWidthRef.current, minWidthPercent);
      widthRef.current = next;
      setWidth(next);
      setLocalStorageItem(storageKey, String(next));
      return;
    }
    lastNonFullWidthRef.current = width;
    widthRef.current = 100;
    setWidth(100);
    setLocalStorageItem(storageKey, "100");
  };

  return (
    <SheetContent
      side="right"
      className={cn(
        "data-[side=right]:w-full data-[side=right]:sm:w-(--sheet-width) data-[side=right]:sm:min-w-[min(720px,100%)] data-[side=right]:sm:max-w-none",
        className,
      )}
      style={{ "--sheet-width": `${width}%` } as React.CSSProperties}
      {...props}
    >
      {children}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize drawer"
        data-slot="sheet-resize-handle"
        className="absolute inset-y-0 left-0 hidden w-1.5 cursor-col-resize touch-none select-none hover:bg-border sm:block"
        onPointerDown={onPointerDown}
      />
      <Button
        variant="ghost"
        size="icon-sm"
        className="absolute top-4 right-14 hidden sm:inline-flex"
        aria-label={isFull ? "Collapse drawer" : "Expand drawer"}
        onClick={toggle}
      >
        {isFull ? <Minimize2Icon /> : <Maximize2Icon />}
      </Button>
    </SheetContent>
  );
}

export { ResizableSheetContent };
