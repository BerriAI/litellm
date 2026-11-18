"use client";

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from "react";
import { Command as CommandPrimitive } from "cmdk";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { FileText, KeyRound, LoaderCircle, Search } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Badge } from "@/components/ui/badge";
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";
import { InputGroupAddon } from "@/components/ui/input-group";
import { useTerminology } from "@/contexts/TerminologyContext";
import i18n from "@/i18n/i18n";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { keyDetailHref } from "@/utils/entityLinks";
import { routeSegmentForPathname, uiHref } from "@/utils/uiHref";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import { useVisibleMenuGroups } from "./useVisibleMenuGroups";
import { flattenNavItems, matchNavItems, paletteScopeForRoute, type PaletteNavItem, type PaletteScope } from "./utils";

type PaletteEntry = {
  id: string;
  title: string;
  subtitle?: ReactNode;
  icon: ReactNode;
  typeLabel: "shell.commandPalette.types.key" | "shell.commandPalette.types.page" | "shell.commandPalette.types.action";
} & (
  | { kind: "key"; key: KeyResponse }
  | { kind: "filter" }
  | { kind: "search-keys" }
  | { kind: "page"; page: PaletteNavItem }
);

interface PaletteGroup {
  heading: string;
  entries: PaletteEntry[];
}

const emptyMessageFor = (scope: PaletteScope, query: string, hasKeyError: boolean): string => {
  if (hasKeyError) return i18n.t("shell.commandPalette.empty.keyError");
  if (scope === "keys" && !query) return i18n.t("shell.commandPalette.empty.noRecentKeys");
  return i18n.t("shell.commandPalette.empty.noResults", { query });
};

// TODO(https://github.com/dip/cmdk/issues/413): remove once cmdk keeps aria-activedescendant on the selected row
function useActiveDescendantOnSelectedRow(inputRef: RefObject<HTMLInputElement | null>) {
  useEffect(() => {
    const input = inputRef.current;
    const root = input?.closest("[cmdk-root]");
    if (!input || !root) return;
    const sync = () => {
      const rowId = root.querySelector('[cmdk-item][aria-selected="true"]')?.id ?? null;
      if (input.getAttribute("aria-activedescendant") === rowId) return;
      if (rowId) input.setAttribute("aria-activedescendant", rowId);
      else input.removeAttribute("aria-activedescendant");
    };
    const observer = new MutationObserver(sync);
    observer.observe(root, {
      subtree: true,
      childList: true,
      attributeFilter: ["aria-selected", "aria-activedescendant"],
    });
    sync();
    return () => observer.disconnect();
  }, [inputRef]);
}

export function CommandPalette({ open, setOpen }: { open: boolean; setOpen: (open: boolean) => void }) {
  const { t } = useTranslation();
  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title={t("shell.commandPalette.title")}
      description={t("shell.commandPalette.description")}
    >
      <CommandPaletteContent setOpen={setOpen} />
    </CommandDialog>
  );
}

function CommandPaletteContent({ setOpen }: { setOpen: (open: boolean) => void }) {
  const { t } = useTranslation();
  const pathname = usePathname();
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  useActiveDescendantOnSelectedRow(inputRef);
  const [query, setQuery] = useState("");
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
  const hasKeyError = scope === "keys" && keyResults.isError && keyResults.data === undefined;
  const visibleGroups = useVisibleMenuGroups();
  const term = useTerminology();
  const navItems = useMemo(() => flattenNavItems(visibleGroups, term), [visibleGroups, term]);
  const navMatches = useMemo(
    () => (scope === "global" || trimmedQuery ? matchNavItems(navItems, query) : []),
    [navItems, query, scope, trimmedQuery],
  );
  const groups = useMemo<PaletteGroup[]>(() => {
    if (hasKeyError) return [];

    const pages: PaletteEntry[] = navMatches.map((page) => ({
      id: `page:${page.route}`,
      kind: "page",
      title: page.label,
      subtitle: page.section,
      icon: page.icon ?? <FileText />,
      typeLabel: "shell.commandPalette.types.page",
      page,
    }));
    const pagesGroups = pages.length > 0 ? [{ heading: t("shell.commandPalette.groups.pages"), entries: pages }] : [];

    if (scope === "global") {
      const showSearchKeysAction =
        !trimmedQuery || t("shell.commandPalette.searchKeys.title").toLowerCase().includes(trimmedQuery.toLowerCase());
      const searchKeysAction: PaletteEntry = {
        id: "action:search-keys",
        kind: "search-keys",
        title: t("shell.commandPalette.searchKeys.title"),
        subtitle: t("shell.commandPalette.searchKeys.subtitle"),
        icon: <KeyRound />,
        typeLabel: "shell.commandPalette.types.action",
      };
      return [
        ...(showSearchKeysAction
          ? [{ heading: t("shell.commandPalette.groups.actions"), entries: [searchKeysAction] }]
          : []),
        ...pagesGroups,
      ];
    }

    const keyEntries: PaletteEntry[] = (isLoadingKeys ? [] : keyResults.data?.keys ?? []).map((key) => {
      const teamAlias = key.team_alias || (key.team_id ? teamAliases.get(key.team_id) : undefined);
      const spend = typeof key.spend === "number" && Number.isFinite(key.spend) ? `$${key.spend.toFixed(2)}` : null;
      const details = [teamAlias, spend].filter((detail): detail is string => Boolean(detail));
      return {
        id: `key:${key.token}`,
        kind: "key",
        title: key.key_alias || t("shell.commandPalette.unnamedKey"),
        subtitle: (
          <>
            <span className="font-mono">{key.key_name}</span>
            {details.length > 0 && <span> · {details.join(" · ")}</span>}
          </>
        ),
        icon: <KeyRound />,
        typeLabel: "shell.commandPalette.types.key",
        key,
      };
    });
    const keysGroups =
      keyEntries.length > 0
        ? [
            {
              heading: trimmedQuery
                ? t("shell.commandPalette.groups.keys")
                : t("shell.commandPalette.groups.recentKeys"),
              entries: keyEntries,
            },
          ]
        : [];

    if (!trimmedQuery) return keysGroups;

    const filterAction: PaletteEntry = {
      id: "action:filter-keys",
      kind: "filter",
      title: t("shell.commandPalette.filterKeys.title", { query: trimmedQuery }),
      subtitle: t("shell.commandPalette.filterKeys.subtitle"),
      icon: <Search />,
      typeLabel: "shell.commandPalette.types.action",
    };
    return [
      ...keysGroups,
      { heading: t("shell.commandPalette.groups.actions"), entries: [filterAction] },
      ...pagesGroups,
    ];
  }, [hasKeyError, isLoadingKeys, keyResults.data?.keys, navMatches, scope, t, teamAliases, trimmedQuery]);

  const resultSet = [scope, query, isLoadingKeys].join("\n");
  const [highlight, setHighlight] = useState<{ value: string; resultSet: string } | null>(null);
  const highlightedValue = highlight?.resultSet === resultSet ? highlight.value : groups[0]?.entries[0]?.id ?? "";

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

  const leaveKeysScopeOnBackspace = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Backspace" || scope !== "keys" || query !== "") return;
    event.preventDefault();
    setScope("global");
  };

  return (
    <Command
      label={t("shell.commandPalette.label")}
      shouldFilter={false}
      loop
      value={highlightedValue}
      onValueChange={(value) => setHighlight({ value, resultSet })}
      onKeyDown={leaveKeysScopeOnBackspace}
    >
      <div className="flex items-center *:data-[slot=command-input-wrapper]:flex-1">
        {scope === "keys" && (
          <InputGroupAddon>
            <Badge variant="secondary">
              <KeyRound />
              {t("shell.commandPalette.virtualKeys")}
            </Badge>
          </InputGroupAddon>
        )}
        <CommandInput
          ref={inputRef}
          placeholder={
            scope === "keys" ? t("shell.commandPalette.placeholder.keys") : t("shell.commandPalette.placeholder.global")
          }
          value={query}
          onValueChange={setQuery}
        />
      </div>
      <CommandList onMouseDown={(event) => event.preventDefault()}>
        {isLoadingKeys && !hasKeyError ? (
          <CommandPrimitive.Loading
            label={t("shell.commandPalette.searching")}
            className="py-6 text-center text-sm text-muted-foreground"
          >
            <span className="inline-flex items-center gap-2">
              <LoaderCircle className="size-4 animate-spin" />
              {t("shell.commandPalette.searching")}
            </span>
          </CommandPrimitive.Loading>
        ) : (
          <CommandEmpty>{emptyMessageFor(scope, trimmedQuery, hasKeyError)}</CommandEmpty>
        )}
        {groups.map((group) => (
          <CommandGroup key={group.heading} heading={group.heading}>
            {group.entries.map((entry) => (
              <CommandItem key={entry.id} value={entry.id} onSelect={() => activate(entry)}>
                {entry.icon}
                <span className="flex min-w-0 items-baseline gap-2">
                  <span className="max-w-full shrink-0 truncate">{entry.title}</span>
                  {entry.subtitle && <span className="truncate text-xs text-muted-foreground">{entry.subtitle}</span>}
                </span>
                <CommandShortcut>{t(entry.typeLabel)}</CommandShortcut>
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </Command>
  );
}
