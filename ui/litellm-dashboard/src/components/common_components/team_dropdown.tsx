import React, { useMemo, useRef, useState } from "react";
import { ChevronsUpDown, Loader2, X } from "lucide-react";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "../key_team_helpers/key_list";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface TeamDropdownProps {
  value?: string;
  onChange?: (value: string) => void;
  /** Callback with the full Team object (or null on clear). */
  onTeamSelect?: (team: Team | null) => void;
  disabled?: boolean;
  /** Filter teams by organization. */
  organizationId?: string | null;
  pageSize?: number;
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

const TeamDropdown: React.FC<TeamDropdownProps> = ({
  value,
  onChange,
  onTeamSelect,
  disabled,
  organizationId,
  pageSize = 20,
}) => {
  const [open, setOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useDebouncedState("", {
    wait: DEBOUNCE_MS,
  });
  const listRef = useRef<HTMLDivElement>(null);

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useInfiniteTeams(pageSize, debouncedSearch || undefined, organizationId);

  const teams = useMemo(() => {
    if (!data?.pages) return [];
    const seen = new Set<string>();
    const result: Team[] = [];
    for (const page of data.pages) {
      for (const team of page.teams) {
        if (seen.has(team.team_id)) continue;
        seen.add(team.team_id);
        result.push(team);
      }
    }
    return result;
  }, [data]);

  const selectedTeam = value
    ? teams.find((t) => t.team_id === value)
    : undefined;

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const scrollRatio =
      (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (scrollRatio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  };

  const handleSearch = (val: string) => {
    setSearchInput(val);
    setDebouncedSearch(val);
  };

  const select = (teamId: string) => {
    onChange?.(teamId);
    if (onTeamSelect) {
      const team = teams.find((t) => t.team_id === teamId) ?? null;
      onTeamSelect(team);
    }
    setOpen(false);
  };

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onChange?.("");
    onTeamSelect?.(null);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          className="w-full justify-between"
          data-testid="team-dropdown"
        >
          {selectedTeam ? (
            <span className="truncate">
              <span className="font-medium">{selectedTeam.team_alias}</span>{" "}
              <span className="text-muted-foreground">
                ({selectedTeam.team_id})
              </span>
            </span>
          ) : (
            <span className="text-muted-foreground">
              Search or select a team
            </span>
          )}
          <div className="ml-2 flex items-center gap-1 shrink-0">
            {value && (
              <span
                role="button"
                tabIndex={0}
                onClick={clear}
                className="text-muted-foreground hover:text-foreground"
                aria-label="Clear"
              >
                <X className="h-3.5 w-3.5" />
              </span>
            )}
            <ChevronsUpDown className="h-4 w-4 opacity-50" />
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[var(--radix-popover-trigger-width)] p-0"
        align="start"
      >
        <div className="p-2 border-b">
          <Input
            autoFocus
            placeholder="Search teams…"
            value={searchInput}
            onChange={(e) => handleSearch(e.target.value)}
            className="h-8"
          />
        </div>
        <div
          ref={listRef}
          onScroll={handleScroll}
          className="max-h-60 overflow-y-auto p-1"
        >
          {isLoading && teams.length === 0 ? (
            <div className="flex items-center justify-center py-6 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          ) : teams.length === 0 ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              No teams found
            </div>
          ) : (
            teams.map((team) => (
              <button
                key={team.team_id}
                type="button"
                onClick={() => select(team.team_id)}
                className={cn(
                  "w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent",
                  value === team.team_id && "bg-accent",
                )}
              >
                <span className="font-medium">{team.team_alias}</span>{" "}
                <span className="text-muted-foreground">
                  ({team.team_id})
                </span>
              </button>
            ))
          )}
          {isFetchingNextPage && (
            <div className="flex items-center justify-center py-2">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default TeamDropdown;
