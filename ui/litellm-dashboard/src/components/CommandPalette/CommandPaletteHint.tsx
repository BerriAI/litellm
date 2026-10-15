"use client";

import { Search, X } from "lucide-react";
import { z } from "zod";
import { storageKey } from "@/lib/storage";
import type { PaletteScope } from "./utils";
import { useCommandPalette } from "./CommandPaletteProvider";
import { useClientMounted, useShortcutLabel } from "./useShortcutLabel";

export const COMMAND_PALETTE_HINT_KEY = storageKey("local", "litellmCommandPaletteHintSeen", z.boolean(), false);

interface CommandPaletteHintProps {
  scope: PaletteScope;
  onDismiss: () => void;
}

export function CommandPaletteHint({ scope, onDismiss }: CommandPaletteHintProps) {
  const { setOpen } = useCommandPalette();
  const shortcut = useShortcutLabel();
  const mounted = useClientMounted();

  if (!mounted) return null;

  return (
    <div className="fixed bottom-4 left-1/2 z-overlay hidden -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-background/80 px-3 py-1.5 text-xs text-muted-foreground shadow-sm backdrop-blur animate-in fade-in slide-in-from-bottom-2 duration-300 md:flex">
      <button type="button" className="flex items-center gap-1.5 hover:text-foreground" onClick={() => setOpen(true)}>
        <Search className="size-3.5" aria-hidden="true" />
        <span>Press</span>
        <kbd className="rounded-sm border border-border border-b-2 bg-muted px-[3px] font-mono text-xs text-muted-foreground">
          {shortcut}
        </kbd>
        <span>{scope === "keys" ? "to search keys" : "to search or jump to a page"}</span>
      </button>
      <button
        type="button"
        className="ml-0.5 flex size-5 items-center justify-center rounded-full hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label="Dismiss search hint"
        onClick={onDismiss}
      >
        <X className="size-3" aria-hidden="true" />
      </button>
    </div>
  );
}
