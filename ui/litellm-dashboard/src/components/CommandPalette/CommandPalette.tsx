"use client";

import { useEffect, useMemo, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { FileText, KeyRound, LoaderCircle, Search } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { menuGroups } from "@/components/leftnav";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { keyDetailHref } from "@/utils/entityLinks";
import { routeSegmentForPathname, uiHref } from "@/utils/uiHref";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import { CommandPaletteResultRow, type CommandPaletteRowData } from "./CommandPaletteResultRow";
import { flattenNavItems, matchNavItems, paletteScopeForRoute, type PaletteNavItem, type PaletteScope } from "./utils";

type PaletteEntry = CommandPaletteRowData &
  (
    | { kind: "key"; key: KeyResponse }
    | { kind: "filter" }
    | { kind: "search-keys" }
    | { kind: "page"; page: PaletteNavItem }
  );

interface PaletteGroup {
  label: string;
  entries: PaletteEntry[];
  emptyMessage?: string;
}

const shortcutChipClass =
  "rounded-sm border border-border border-b-2 bg-muted px-[3px] font-mono text-muted-foreground";

export function CommandPalette({ open, setOpen }: { open: boolean; setOpen: (open: boolean) => void }) {
  if (!open) return null;
  return <CommandPaletteDialog setOpen={setOpen} />;
}

function CommandPaletteDialog({ setOpen }: { setOpen: (open: boolean) => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const { userRole } = useAuthorized();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);
  const [scope, setScope] = useState<PaletteScope>(() => paletteScopeForRoute(routeSegmentForPathname(pathname)));
  const [debouncedQuery] = useDebouncedValue(query, { wait: DEBOUNCE_WAIT_MS });
  const trimmedQuery = query.trim();
  const debouncedSearch = debouncedQuery.trim();
  const keyListOptions = {
    search: debouncedSearch || undefined,
    sortBy: "created_at",
    sortOrder: "desc",
    expand: "user",
  };
  const keyResults = useKeys(1, 8, keyListOptions, { enabled: scope === "keys" });
  const navItems = useMemo(() => flattenNavItems(menuGroups, userRole), [userRole]);
  const navMatches = useMemo(
    () => (scope === "global" || trimmedQuery ? matchNavItems(navItems, query) : []),
    [navItems, query, scope, trimmedQuery],
  );
  const groups = useMemo<PaletteGroup[]>(() => {
    const pages: PaletteEntry[] = navMatches.map((page) => ({
      id: `page:${page.route}`,
      kind: "page",
      title: page.label,
      subtitle: page.section,
      icon: page.icon ?? <FileText className="size-4" />,
      typeLabel: "Page",
      page,
    }));

    if (scope === "global") {
      return [
        {
          label: "Actions",
          entries: [
            {
              id: "action:search-keys",
              kind: "search-keys",
              title: "Search virtual keys",
              subtitle: "Find a key by alias or ID",
              icon: <KeyRound className="size-4" />,
              typeLabel: "Action",
            },
          ],
        },
        { label: "Pages", entries: pages, emptyMessage: "No results" },
      ];
    }

    const keyEntries: PaletteEntry[] = (keyResults.data?.keys ?? []).map((key) => {
      const team = key.team_alias || key.team_id;
      const spend = typeof key.spend === "number" && Number.isFinite(key.spend) ? `$${key.spend.toFixed(2)}` : null;
      const details = [team, spend].filter((detail): detail is string => Boolean(detail));
      const subtitle: ReactNode = (
        <>
          <span className="font-mono">{key.key_name}</span>
          {details.length > 0 && <span> · {details.join(" · ")}</span>}
        </>
      );
      return {
        id: `key:${key.token}`,
        kind: "key",
        title: key.key_alias || "Unnamed key",
        subtitle,
        icon: <KeyRound className="size-4" />,
        typeLabel: "Key",
        key,
      };
    });
    const keysGroup: PaletteGroup = {
      label: trimmedQuery ? "Keys" : "Recent keys",
      entries: keyEntries,
      emptyMessage: trimmedQuery ? `No keys match “${trimmedQuery}”` : "No results",
    };

    if (!trimmedQuery) return [keysGroup];

    const filterEntry: PaletteEntry = {
      id: "action:filter-keys",
      kind: "filter",
      title: `Filter the keys table for “${trimmedQuery}”`,
      subtitle: "Open Virtual Keys with this search applied",
      icon: <Search className="size-4" />,
      typeLabel: "Action",
    };

    return [
      keysGroup,
      { label: "Actions", entries: [filterEntry] },
      { label: "Pages", entries: pages, emptyMessage: "No results" },
    ];
  }, [keyResults.data?.keys, navMatches, scope, trimmedQuery]);
  const selectableEntries = useMemo(() => groups.flatMap((group) => group.entries), [groups]);
  const activeId =
    activeIndex >= 0 && activeIndex < selectableEntries.length ? `command-palette-option-${activeIndex}` : undefined;
  const isSearching = scope === "keys" && keyResults.isFetching && keyResults.data === undefined;
  const hasKeyError = scope === "keys" && keyResults.isError && keyResults.data === undefined;

  useEffect(() => {
    if (activeId) document.getElementById(activeId)?.scrollIntoView?.({ block: "nearest" });
  }, [activeId]);

  const activate = (entry: PaletteEntry) => {
    switch (entry.kind) {
      case "key":
        router.push(keyDetailHref(entry.key.token));
        setOpen(false);
        return;
      case "filter":
        router.push(`${uiHref("api-keys")}?key_search=${encodeURIComponent(trimmedQuery)}`);
        setOpen(false);
        return;
      case "search-keys":
        setScope("keys");
        setActiveIndex(-1);
        return;
      case "page":
        router.push(uiHref(entry.page.route));
        setOpen(false);
        return;
    }
  };

  const handleInputKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Backspace" && scope === "keys" && query === "") {
      event.preventDefault();
      setScope("global");
      setActiveIndex(-1);
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      const activeEntry = selectableEntries[activeIndex];
      if (activeEntry) activate(activeEntry);
      return;
    }

    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    if (selectableEntries.length === 0) return;
    setActiveIndex((current) => {
      if (event.key === "ArrowDown") return current < 0 || current >= selectableEntries.length - 1 ? 0 : current + 1;
      return current <= 0 || current >= selectableEntries.length ? selectableEntries.length - 1 : current - 1;
    });
  };

  return (
    <Dialog open onOpenChange={setOpen}>
      <DialogContent
        showCloseButton={false}
        overlayClassName="bg-black/35 backdrop-blur-sm"
        className="top-[15vh] max-h-[70vh] translate-y-0 gap-0 overflow-hidden rounded-xl border border-border p-0 shadow-2xl sm:max-w-[640px]"
      >
        <DialogTitle className="sr-only">Command palette</DialogTitle>
        <div className="flex h-14 items-center gap-3 border-b border-border px-4">
          <Search className="size-4 shrink-0 text-muted-foreground" />
          {scope === "keys" && (
            <span className="flex shrink-0 items-center gap-1.5 rounded-md bg-muted px-2 py-1 text-xs font-medium">
              <KeyRound className="size-3.5" />
              Virtual Keys
            </span>
          )}
          <input
            autoFocus
            type="text"
            role="combobox"
            aria-label="Search"
            aria-expanded="true"
            aria-controls="command-palette-results"
            aria-activedescendant={activeId}
            aria-autocomplete="list"
            className="h-full min-w-0 flex-1 border-0 bg-transparent text-base outline-none placeholder:text-muted-foreground focus:ring-0"
            placeholder={scope === "keys" ? "Search keys by alias or ID…" : "Search pages and actions…"}
            value={query}
            onChange={(event) => {
              setQuery(event.currentTarget.value);
              setActiveIndex(-1);
            }}
            onKeyDown={handleInputKeyDown}
          />
        </div>
        <div id="command-palette-results" role="listbox" className="max-h-[400px] overflow-y-auto p-2">
          {isSearching && (
            <div className="flex h-20 items-center justify-center gap-2 text-sm text-muted-foreground">
              <LoaderCircle className="size-4 animate-spin" />
              Searching…
            </div>
          )}
          {hasKeyError && (
            <div className="flex h-20 items-center justify-center text-sm text-muted-foreground">
              Unable to search keys
            </div>
          )}
          {!isSearching &&
            !hasKeyError &&
            groups.map((group, groupIndex) => {
              const firstOptionIndex = groups
                .slice(0, groupIndex)
                .reduce((count, previousGroup) => count + previousGroup.entries.length, 0);
              return (
                <div key={group.label} role="group" aria-label={group.label}>
                  <div
                    role="presentation"
                    className="px-2 py-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground"
                  >
                    {group.label}
                  </div>
                  {group.entries.length === 0 ? (
                    <div className="px-2 py-3 text-sm text-muted-foreground">{group.emptyMessage}</div>
                  ) : (
                    group.entries.map((entry, entryIndex) => {
                      const currentIndex = firstOptionIndex + entryIndex;
                      return (
                        <CommandPaletteResultRow
                          key={entry.id}
                          item={entry}
                          optionId={`command-palette-option-${currentIndex}`}
                          active={activeIndex === currentIndex}
                          onActivate={() => activate(entry)}
                          onHover={() => setActiveIndex(currentIndex)}
                        />
                      );
                    })
                  )}
                </div>
              );
            })}
        </div>
        <div className="flex h-10 items-center justify-between border-t border-border bg-muted/30 px-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="font-medium text-foreground">LiteLLM</span>
            <span>{scope === "keys" ? "Virtual Keys" : "All pages"}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <kbd className={shortcutChipClass}>↵</kbd> Open
            </span>
            <span className="flex items-center gap-1">
              <kbd className={shortcutChipClass}>↑↓</kbd> Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className={shortcutChipClass}>esc</kbd> Close
            </span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
