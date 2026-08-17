import React, { useMemo, useState, type UIEvent } from "react";
import { Loader2 } from "lucide-react";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxClear,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { Team } from "../key_team_helpers/key_list";

interface TeamMultiSelectProps {
  value?: string[];
  onChange?: (value: string[]) => void;
  disabled?: boolean;
  organizationId?: string | null;
  pageSize?: number;
  placeholder?: string;
}

const SCROLL_THRESHOLD = 0.8;

const TeamMultiSelect: React.FC<TeamMultiSelectProps> = ({
  value = [],
  onChange,
  disabled,
  organizationId,
  pageSize = 20,
  placeholder = "Search teams by alias...",
}) => {
  const anchor = useComboboxAnchor();
  const [search, setSearch] = useState("");
  const debouncedSetSearch = useDebouncedCallback(setSearch, { wait: DEBOUNCE_WAIT_MS });

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteTeams(
    pageSize,
    search || undefined,
    organizationId,
  );

  const teamById = useMemo(
    () =>
      new Map<string, Team>(
        (data?.pages ?? []).flatMap((page) => page.teams).map((team) => [team.team_id, team] as const),
      ),
    [data],
  );
  const teamIds = useMemo(() => Array.from(teamById.keys()), [teamById]);

  const aliasOf = (teamId: string) => teamById.get(teamId)?.team_alias ?? teamId;

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    if (target.scrollHeight === 0) return;
    const scrollRatio = (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (scrollRatio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  };

  return (
    <Combobox
      multiple
      items={teamIds}
      value={value}
      onValueChange={(next: string[]) => onChange?.(next)}
      filter={null}
      onInputValueChange={debouncedSetSearch}
      disabled={disabled}
    >
      <ComboboxChips render={<div ref={anchor} />} className="w-full" aria-busy={isLoading}>
        <ComboboxValue>
          {(selected: string[]) =>
            selected.map((teamId) => (
              <ComboboxChip key={teamId} aria-label={aliasOf(teamId)}>
                {aliasOf(teamId)}
              </ComboboxChip>
            ))
          }
        </ComboboxValue>
        <ComboboxChipsInput placeholder={placeholder} aria-label={placeholder} disabled={disabled} />
        {value.length > 0 && <ComboboxClear aria-label="Clear all teams" disabled={disabled} />}
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>
          {isLoading ? <Loader2 className="size-4 animate-spin text-muted-foreground" /> : "No teams found"}
        </ComboboxEmpty>
        <ComboboxList onScroll={handleScroll}>
          {(teamId: string) => (
            <ComboboxItem key={teamId} value={teamId}>
              <span className="font-medium">{aliasOf(teamId)}</span>{" "}
              <span className="text-muted-foreground">({teamId})</span>
            </ComboboxItem>
          )}
        </ComboboxList>
        {isFetchingNextPage && (
          <div className="flex justify-center py-2">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        )}
      </ComboboxContent>
    </Combobox>
  );
};

export default TeamMultiSelect;
