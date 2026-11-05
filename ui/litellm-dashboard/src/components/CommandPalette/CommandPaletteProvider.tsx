"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useHotkeys } from "react-hotkeys-hook";
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

const isInsideDialog = (event: KeyboardEvent): boolean =>
  event.target instanceof Element && event.target.closest("[role='dialog'], [role='alertdialog']") !== null;

export function useCommandPalette(): CommandPaletteContextValue {
  const context = useContext(CommandPaletteContext);
  if (!context) throw new Error("useCommandPalette must be used within CommandPaletteProvider");
  return context;
}

export function CommandPaletteProvider({ children }: { children: ReactNode }) {
  const [open, setOpenState] = useState(false);
  const [hintSeen, setHintSeen] = useStoredValue(COMMAND_PALETTE_HINT_KEY);
  const [hintDismissedInMemory, setHintDismissedInMemory] = useState(false);
  const pathname = usePathname();
  const scope = paletteScopeForRoute(routeSegmentForPathname(pathname));
  const markHintSeen = useCallback(() => {
    setHintDismissedInMemory(true);
    setHintSeen(true);
  }, [setHintSeen]);
  const setOpen = useCallback(
    (nextOpen: boolean) => {
      if (nextOpen) markHintSeen();
      setOpenState(nextOpen);
    },
    [markHintSeen],
  );
  const toggle = useCallback(() => setOpen(!open), [open, setOpen]);
  const value = useMemo(() => ({ open, setOpen, toggle }), [open, setOpen, toggle]);
  const shouldShowHint = !hintSeen && !hintDismissedInMemory && !open;

  const hotkeyOptions = {
    enableOnFormTags: true,
    preventDefault: true,
    useKey: true,
    ignoreEventWhen: (event: KeyboardEvent) => !open && isInsideDialog(event),
  };
  useHotkeys("mod+k", toggle, hotkeyOptions);

  return (
    <CommandPaletteContext.Provider value={value}>
      {children}
      <CommandPalette open={open} setOpen={setOpen} />
      {shouldShowHint && <CommandPaletteHint scope={scope} onDismiss={markHintSeen} />}
    </CommandPaletteContext.Provider>
  );
}
