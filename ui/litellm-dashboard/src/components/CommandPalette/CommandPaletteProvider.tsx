"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useStoredValue } from "@/lib/storage";
import { routeSegmentForPathname } from "@/utils/uiHref";
import { CommandPalette } from "./CommandPalette";
import { CommandPaletteHint, COMMAND_PALETTE_HINT_KEY } from "./CommandPaletteHint";
import { paletteScopeForRoute } from "./utils";

interface CommandPaletteContextValue {
  open: boolean;
  setOpen: (open: boolean) => void;
  toggle: () => void;
}

const CommandPaletteContext = createContext<CommandPaletteContextValue | null>(null);

export function useCommandPalette(): CommandPaletteContextValue {
  const context = useContext(CommandPaletteContext);
  if (!context) throw new Error("useCommandPalette must be used within CommandPaletteProvider");
  return context;
}

export function CommandPaletteProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [hintSeen, setHintSeen] = useStoredValue(COMMAND_PALETTE_HINT_KEY);
  const [hintDismissedInMemory, setHintDismissedInMemory] = useState(false);
  const pathname = usePathname();
  const toggle = useCallback(() => setOpen((current) => !current), []);
  const value = useMemo(() => ({ open, setOpen, toggle }), [open, toggle]);
  const scope = paletteScopeForRoute(routeSegmentForPathname(pathname));
  const markHintSeen = useCallback(() => {
    setHintDismissedInMemory(true);
    setHintSeen(true);
  }, [setHintSeen]);
  const shouldShowHint = !hintSeen && !hintDismissedInMemory && !open;

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k" && !event.shiftKey && !event.altKey) {
        event.preventDefault();
        toggle();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [toggle]);

  useEffect(() => {
    if (open) markHintSeen();
  }, [markHintSeen, open]);

  return (
    <CommandPaletteContext.Provider value={value}>
      {children}
      <CommandPalette open={open} setOpen={setOpen} />
      {shouldShowHint && <CommandPaletteHint scope={scope} onDismiss={markHintSeen} />}
    </CommandPaletteContext.Provider>
  );
}
