"use client";

import { cn } from "@/lib/cva.config";
import { Search } from "lucide-react";
import { usePathname } from "next/navigation";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { routeSegmentForPathname } from "@/utils/uiHref";
import { paletteScopeForRoute } from "./utils";
import { useCommandPalette } from "./CommandPaletteProvider";
import { useShortcutLabel } from "./useShortcutLabel";

export function CommandPaletteTrigger({ mobile = false, className }: { mobile?: boolean; className?: string }) {
  const { t } = useTranslation();
  const pathname = usePathname();
  const { open, setOpen } = useCommandPalette();
  const shortcut = useShortcutLabel();
  const isKeysRoute = paletteScopeForRoute(routeSegmentForPathname(pathname)) === "keys";
  const label = isKeysRoute ? t("shell.commandPaletteTrigger.keys") : t("shell.commandPaletteTrigger.global");

  if (mobile) {
    return (
      <Button
        variant="ghost"
        size="icon"
        className={cn("text-muted-foreground", className ?? "size-11 md:hidden")}
        aria-label={t("shell.commandPaletteTrigger.search")}
        aria-expanded={open}
        onClick={() => setOpen(true)}
      >
        <Search />
      </Button>
    );
  }

  return (
    <Button
      variant="ghost"
      className={cn(
        "justify-between bg-muted/50 font-normal text-muted-foreground hover:bg-muted",
        className ?? "hidden w-64 md:flex lg:w-80",
      )}
      aria-label={label}
      aria-expanded={open}
      onClick={() => setOpen(true)}
    >
      <span className="flex min-w-0 items-center gap-2">
        <Search />
        <span className="truncate">{label}</span>
      </span>
      <Kbd className="bg-background/70 font-normal">{shortcut}</Kbd>
    </Button>
  );
}
