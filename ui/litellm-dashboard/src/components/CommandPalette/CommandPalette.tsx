"use client";

import { useEffect, useMemo, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { FileText, KeyRound, LoaderCircle, Search } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { keyDetailHref } from "@/utils/entityLinks";
import { routeSegmentForPathname, uiHref } from "@/utils/uiHref";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import { CommandPaletteResultRow, type CommandPaletteRowData } from "./CommandPaletteResultRow";
import { useVisibleMenuGroups } from "./useVisibleMenuGroups";
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
  isLoading?: boolean;
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
  const [query, setQuery] = useState("");
  const [activeSelection, setActiveSelection] = useState<{
    id: string;
    query: string;
    scope: PaletteScope;
  } | null>(null);
  const [scope, setScope] = useState<PaletteScope>(() => paletteScopeForRoute(routeSegmentForPathname(pathname)));
  const [debouncedQuery] = useDebouncedValue(query, { wait: DEBOUNCE_WAIT_MS });
  const trimmedQuery = query.trim();
  const debouncedSearch = debouncedQuery.trim();
  const { data: teams } = useTeams({ enabled: false });
  const teamAliases = useMemo(
    () => new Map((teams ?? []).map((team) => [team.team_id, team.team_alias] as const)),
    [teams],
  );
  const keyListOptions = {
    search: debouncedSearch || undefined,
    sortBy: "created_at",
    sortOrder: "desc",
    expand: "user",
  };
  const keyResults = useKeys(1, 8, keyListOptions, { enabled: scope === "keys" });
  const isPendingQuery = trimmedQuery !== debouncedSearch;
  const hasPlaceholderKeyResults = keyResults.isPlaceholderData;
  const isInitialKeyLoad = keyResults.isFetching && !keyResults.data;
  const isLoadingKeys = scope === "keys" && (isPendingQuery || hasPlaceholderKeyResults || isInitialKeyLoad);
  const visibleGroups = useVisibleMenuGroups();
  const navItems = useMemo(() => flattenNavItems(visibleGroups), [visibleGroups]);
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
      const searchKeysAction = !trimmedQuery || "search virtual keys".includes(trimmedQuery.toLowerCase());
      return [
        ...(searchKeysAction
          ? [
              {
                label: "Actions",
                entries: [
                  {
                    id: "action:search-keys",
                    kind: "search-keys" as const,
                    title: "Search virtual keys",
                    subtitle: "Find a key by alias or ID",
                    icon: <KeyRound className="size-4" />,
                    typeLabel: "Action" as const,
                  },
                ],
              },
            ]
          : []),
        ...(pages.length > 0 ? [{ label: "Pages", entries: pages }] : []),
      ];
    }

    const keyEntries: PaletteEntry[] = (isLoadingKeys ? [] : keyResults.data?.keys ?? []).map((key) => {
      const teamAlias = key.team_alias || (key.team_id ? teamAliases.get(key.team_id) : undefined);
      const spend = typeof key.spend === "number" && Number.isFinite(key.spend) ? `$${key.spend.toFixed(2)}` : null;
      const details = [teamAlias, spend].filter((detail): detail is string => Boolean(detail));
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
      emptyMessage: trimmedQuery ? `No keys match “${trimmedQuery}”` : "No recent keys",
      isLoading: isLoadingKeys,
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
      ...(pages.length > 0 ? [{ label: "Pages", entries: pages }] : []),
    ];
  }, [isLoadingKeys, keyResults.data?.keys, navMatches, scope, teamAliases, trimmedQuery]);
  const selectableEntries = useMemo(() => groups.flatMap((group) => group.entries), [groups]);
  const activeEntryId = activeSelection?.query === query && activeSelection.scope === scope ? activeSelection.id : null;
  const selectedIndex = selectableEntries.findIndex((entry) => entry.id === activeEntryId);
  const activeIndex = selectableEntries.length === 0 ? -1 : Math.max(0, selectedIndex);
  const activeOptionId = activeIndex >= 0 ? `command-palette-option-${activeIndex}` : undefined;
  const hasKeyError = scope === "keys" && keyResults.isError && keyResults.data === undefined;

  useEffect(() => {
    if (activeOptionId) document.getElementById(activeOptionId)?.scrollIntoView?.({ block: "nearest" });
  }, [activeOptionId]);

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
        setQuery("");
        setScope("keys");
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
    const currentIndex = activeIndex < 0 ? 0 : activeIndex;
    const nextIndex =
      event.key === "ArrowDown"
        ? (currentIndex + 1) % selectableEntries.length
        : (currentIndex - 1 + selectableEntries.length) % selectableEntries.length;
    setActiveSelection({ id: selectableEntries[nextIndex].id, query, scope });
  };

  const renderGroupContents = (group: PaletteGroup, firstOptionIndex: number): ReactNode => {
    if (group.isLoading) {
      return (
        <div className="flex h-20 items-center justify-center gap-2 text-sm text-muted-foreground">
          <LoaderCircle className="size-4 animate-spin" />
          Searching…
        </div>
      );
    }
    if (group.entries.length === 0)
      return <div className="px-2 py-3 text-sm text-muted-foreground">{group.emptyMessage}</div>;
    return group.entries.map((entry, entryIndex) => {
      const currentIndex = firstOptionIndex + entryIndex;
      return (
        <CommandPaletteResultRow
          key={entry.id}
          item={entry}
          optionId={`command-palette-option-${currentIndex}`}
          active={activeIndex === currentIndex}
          onActivate={() => activate(entry)}
          onHover={() => setActiveSelection({ id: entry.id, query, scope })}
        />
      );
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
            aria-activedescendant={activeOptionId}
            aria-autocomplete="list"
            className="h-full min-w-0 flex-1 border-0 bg-transparent text-[15px] outline-none placeholder:text-muted-foreground focus:ring-0"
            placeholder={scope === "keys" ? "Search keys by alias or ID…" : "Search pages and actions…"}
            value={query}
            onChange={(event) => {
              setQuery(event.currentTarget.value);
            }}
            onKeyDown={handleInputKeyDown}
          />
        </div>
        <div id="command-palette-results" role="listbox" className="max-h-[400px] overflow-y-auto p-2">
          {hasKeyError && (
            <div className="flex h-20 items-center justify-center text-sm text-muted-foreground">
              Unable to search keys
            </div>
          )}
          {!hasKeyError && scope === "global" && trimmedQuery && groups.length === 0 && (
            <div className="px-2 py-3 text-sm text-muted-foreground">No results for “{trimmedQuery}”</div>
          )}
          {!hasKeyError &&
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
                  {renderGroupContents(group, firstOptionIndex)}
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
