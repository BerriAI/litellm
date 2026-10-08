"use client";

import { Search } from "lucide-react";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { routeSegmentForPathname } from "@/utils/uiHref";
import { paletteScopeForRoute } from "./utils";
import { useCommandPalette } from "./CommandPaletteProvider";
import { useShortcutLabel } from "./useShortcutLabel";

export function CommandPaletteTrigger({ mobile = false }: { mobile?: boolean }) {
  const pathname = usePathname();
  const { open, setOpen } = useCommandPalette();
  const shortcut = useShortcutLabel();
  const isKeysRoute = paletteScopeForRoute(routeSegmentForPathname(pathname)) === "keys";
  const label = isKeysRoute ? "Search keys…" : "Search or jump to…";

  if (mobile) {
    return (
      <Button
        variant="ghost"
        size="icon"
        className="size-11 md:hidden"
        aria-label="Search"
        aria-expanded={open}
        onClick={() => setOpen(true)}
      >
        <Search />
      </Button>
    );
  }

  return (
    <button
      type="button"
      className="hidden h-8 w-64 shrink-0 items-center justify-between rounded-lg border border-border bg-muted/40 px-2.5 text-sm text-muted-foreground transition-colors hover:bg-muted lg:w-80 md:flex"
      aria-label={label}
      aria-expanded={open}
      onClick={() => setOpen(true)}
    >
      <span className="flex min-w-0 items-center gap-2">
        <Search className="size-4 shrink-0" />
        <span className="truncate">{label}</span>
      </span>
      <kbd className="ml-2 shrink-0 rounded-sm border border-border border-b-2 bg-muted px-[3px] font-mono text-xs text-muted-foreground">
        {shortcut}
      </kbd>
    </button>
  );
}
