import React, { useMemo, useRef, useState } from "react";
import { Loader2, X } from "lucide-react";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "../key_team_helpers/key_list";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

interface TeamMultiSelectProps {
  value?: string[];
  onChange?: (value: string[]) => void;
  disabled?: boolean;
  organizationId?: string | null;
  pageSize?: number;
  placeholder?: string;
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

const TeamMultiSelect: React.FC<TeamMultiSelectProps> = ({
  value = [],
  onChange,
  disabled,
  organizationId,
  pageSize = 20,
  placeholder = "Search teams by alias...",
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

  const teamMap = useMemo(() => {
    const m = new Map<string, Team>();
    for (const team of teams) m.set(team.team_id, team);
    return m;
  }, [teams]);

  const labelFor = (id: string) => teamMap.get(id)?.team_alias ?? id;

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

  const toggle = (teamId: string) => {
    if (value.includes(teamId)) {
      onChange?.(value.filter((v) => v !== teamId));
    } else {
      onChange?.([...value, teamId]);
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left disabled:opacity-50",
          )}
        >
          {value.length === 0 ? (
            <span className="text-muted-foreground px-1">{placeholder}</span>
          ) : (
            value.map((id) => (
              <Badge
                key={id}
                variant="secondary"
                className="gap-1 inline-flex items-center"
              >
                {labelFor(id)}
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange?.(value.filter((s) => s !== id));
                  }}
                  className="inline-flex items-center"
                  aria-label={`Remove ${labelFor(id)}`}
                >
                  <X size={12} />
                </span>
              </Badge>
            ))
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] p-0"
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
            teams.map((team) => {
              const selected = value.includes(team.team_id);
              return (
                <button
                  key={team.team_id}
                  type="button"
                  onClick={() => toggle(team.team_id)}
                  className={cn(
                    "w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent",
                    selected && "bg-accent",
                  )}
                >
                  <span className="font-medium">{team.team_alias}</span>{" "}
                  <span className="text-muted-foreground">
                    ({team.team_id})
                  </span>
                </button>
              );
            })
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

export default TeamMultiSelect;
