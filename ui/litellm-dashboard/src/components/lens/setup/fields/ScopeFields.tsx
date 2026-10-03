"use client";

import { useState, type ReactNode } from "react";
import {
  Combobox,
  ComboboxInput,
  ComboboxContent,
  ComboboxList,
  ComboboxItem,
  ComboboxEmpty,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { type Sample, type Settings, type ActivitySelection } from "../../model/types";

import { MetadataFilters } from "./MetadataFilters";
const selectClass = "h-9 w-full rounded-md border border-input bg-background px-3 text-sm";
export function ScopeFields({
  value,
  onChange,
  nameField,
  names,
  selectedName,
  selectName,
  agents,
  attributes,
  keys,
  id,
}: {
  value: ActivitySelection;
  onChange: (value: ActivitySelection) => void;
  nameField?: ReactNode;
  names: string[];
  selectedName: string | undefined;
  selectName: (name: string) => void;
  agents: import("@tanstack/react-query").UseQueryResult<string[], Error>;
  attributes: NonNullable<Sample["executions"][number]["metadata"]>;
  keys: string[];
  id: string;
}) {
  const filters = value.filters ?? [];
  const hasFilters = !!value.filters?.length || !!value.team_id;
  const [advanced, setAdvanced] = useState(hasFilters || !!value.service || value.source !== "traces");
  const changeSource = (source: Settings["source"]) => {
    const selection = { ...value, source, service: "", agent_name: "", filters: [], execution_ids: [] };
    onChange(selection);
  };
  return (
    <>
      {nameField}
      <label className="grid gap-2 text-sm font-medium">
        {value.source === "requests" ? "Model group (optional)" : "Agent (optional)"}
        <Combobox
          items={names}
          value={selectedName || null}
          inputValue={selectedName ?? ""}
          onInputValueChange={selectName}
          onValueChange={(name) => selectName(name ?? "")}
        >
          <ComboboxInput
            aria-label={value.source === "requests" ? "Model group (optional)" : "Agent (optional)"}
            placeholder={value.source === "requests" ? "All model groups" : "All agents and activity"}
            showClear={!!selectedName}
            className="w-full h-9"
          />
          <ComboboxContent>
            <ComboboxEmpty>
              {agents.isFetching ? "Loading agents…" : "No matches. You can enter a recorded name."}
            </ComboboxEmpty>
            <ComboboxList>
              {(name: string) => (
                <ComboboxItem key={name} value={name}>
                  {name}
                </ComboboxItem>
              )}
            </ComboboxList>
          </ComboboxContent>
        </Combobox>
      </label>
      {value.source !== "requests" && agents.isError && (
        <p role="alert" className="text-sm text-destructive">
          Could not load agents.{" "}
          <button type="button" className="underline" onClick={() => void agents.refetch()}>
            Retry
          </button>
        </p>
      )}
      <details open={advanced} onToggle={(event) => setAdvanced(event.currentTarget.open)} className="group">
        <summary className="cursor-pointer text-sm font-medium">
          Advanced filters{filters.length ? ` (${filters.length})` : ""}
        </summary>
        <div className="mt-4 space-y-4">
          {value.source !== "requests" && (
            <label className="grid gap-2 text-sm">
              Application (optional)
              <Input
                value={value.service ?? ""}
                placeholder="All applications"
                onChange={(event) => onChange({ ...value, service: event.target.value, execution_ids: [] })}
              />
            </label>
          )}
          <label className="grid gap-2 text-sm">
            Activity type
            <select
              className={selectClass}
              value={value.source}
              onChange={(e) => changeSource(e.target.value as Settings["source"])}
            >
              <option value="traces">Agent traces</option>
              <option value="requests">LLM requests</option>
              <option value="both">Traces and LLM requests</option>
            </select>
          </label>
          <p className="text-xs leading-5 text-muted-foreground">
            Match any recorded metadata, such as a user ID, environment, or tag. All conditions must match.
          </p>
          <MetadataFilters value={value} onChange={onChange} attributes={attributes} keys={keys} id={id} />
          <label className="grid gap-2 text-sm">
            Team ID (optional)
            <Input
              value={value.team_id ?? ""}
              placeholder="All accessible teams"
              onChange={(e) => onChange({ ...value, team_id: e.target.value })}
            />
          </label>
        </div>
      </details>
    </>
  );
}
