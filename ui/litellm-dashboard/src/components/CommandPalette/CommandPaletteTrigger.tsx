"use client";

import { Search } from "lucide-react";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
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
    <Button
      variant="outline"
      className="hidden w-64 justify-between md:flex lg:w-80"
      aria-label={label}
      aria-expanded={open}
      onClick={() => setOpen(true)}
    >
      <span className="flex min-w-0 items-center gap-2">
        <Search />
        <span className="truncate">{label}</span>
      </span>
      <Kbd>{shortcut}</Kbd>
    </Button>
  );
}
