"use client";

import { useState, type ReactNode } from "react";
import { Plus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxCollection,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxGroup,
  ComboboxInput,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
  ComboboxTrigger,
} from "@/components/ui/combobox";
import { InputGroupButton } from "@/components/ui/input-group";

export type PickerGroup = { value: string; items: readonly string[] };

export const MODEL_PICKER_GROUPS: readonly PickerGroup[] = [
  {
    value: "Anthropic",
    items: ["anthropic-haiku-4-5", "anthropic-sonnet-4-6", "anthropic-sonnet-5", "anthropic-opus-4-8"],
  },
  { value: "OpenAI", items: ["gpt-5.5", "gpt-5.5-mini", "o5"] },
  { value: "Bedrock", items: ["bedrock-converse-sonnet-4-6", "bedrock-invoke-opus-4-8"] },
  {
    value: "Vertex AI",
    items: ["vertex-gemini-3-pro", "vertex-ai-claude-sonnet-4-6-europe-west4-provisioned-throughput"],
  },
];

export const MCP_GROUPS: readonly PickerGroup[] = [
  { value: "MCP servers", items: ["All MCP servers", "github", "linear", "slack", "sentry", "notion"] },
  { value: "Toolsets", items: ["support-agent", "release-bot"] },
];

export const VECTOR_STORE_GROUPS: readonly PickerGroup[] = [
  { value: "Vector stores", items: ["support-docs", "eng-wiki"] },
];

export const GUARDRAIL_GROUPS: readonly PickerGroup[] = [
  { value: "Guardrails", items: ["pii-mask", "prompt-injection", "toxicity", "secret-detection"] },
  { value: "Policies", items: ["eu-data-residency", "hipaa"] },
];

export const SERVER_TOOLS: Readonly<Record<string, readonly string[]>> = {
  github: [
    "create_issue",
    "get_issue",
    "update_issue",
    "list_issues",
    "create_pull_request",
    "get_pull_request",
    "list_pull_requests",
    "merge_pull_request",
    "search_code",
    "get_file_contents",
    "create_branch",
    "list_commits",
  ],
  linear: [
    "create_issue",
    "get_issue",
    "update_issue",
    "search_issues",
    "list_projects",
    "list_teams",
    "create_comment",
  ],
  slack: ["post_message", "list_channels", "get_channel_history", "search_messages", "add_reaction"],
  sentry: ["list_issues", "get_issue", "get_event", "resolve_issue"],
  notion: ["search", "get_page", "create_page", "update_page", "query_database"],
};

export const DEFAULT_MCP: readonly string[] = ["github", "linear"];
export const DEFAULT_GUARDRAILS: readonly string[] = ["pii-mask", "prompt-injection"];

function GroupedList({ mono = false }: { mono?: boolean }) {
  return (
    <ComboboxList>
      {(group: PickerGroup) => (
        <ComboboxGroup key={group.value} items={[...group.items]}>
          <ComboboxLabel>{group.value}</ComboboxLabel>
          <ComboboxCollection>
            {(item: string) => (
              <ComboboxItem key={item} value={item} className={mono ? "font-mono text-xs" : undefined}>
                <span className="min-w-0 truncate">{item}</span>
              </ComboboxItem>
            )}
          </ComboboxCollection>
        </ComboboxGroup>
      )}
    </ComboboxList>
  );
}

export function ModelPicker({ initial, open }: { initial: string; open: boolean }) {
  const [model, setModel] = useState<string | null>(initial);
  const [query, setQuery] = useState("");
  return (
    <Combobox
      items={[...MODEL_PICKER_GROUPS]}
      value={model}
      onValueChange={(value: string | null) => value && setModel(value)}
      inputValue={query}
      onInputValueChange={setQuery}
      defaultOpen={open}
    >
      <ComboboxTrigger render={<InputGroupButton size="xs" className="max-w-72 min-w-0" aria-label="Model" />}>
        <span className="min-w-0 truncate font-mono text-xs font-medium text-foreground">{model}</span>
      </ComboboxTrigger>
      <ComboboxContent side="top" className="w-80">
        <ComboboxInput showTrigger={false} placeholder="Search models" />
        <ComboboxEmpty className="flex-col items-stretch gap-1 px-1 pb-1">
          <span className="py-1">No matching model</span>
          <Button
            variant="ghost"
            size="sm"
            className="min-w-0 justify-start"
            onClick={() => {
              setModel(query);
              setQuery("");
            }}
          >
            <Plus />
            <span className="min-w-0 truncate">
              Use <span className="font-mono">{query}</span>
            </span>
          </Button>
        </ComboboxEmpty>
        <GroupedList mono />
      </ComboboxContent>
    </Combobox>
  );
}

export function CountPicker({
  label,
  icon,
  groups,
  initial,
  open,
}: {
  label: string;
  icon: ReactNode;
  groups: readonly PickerGroup[];
  initial: readonly string[];
  open: boolean;
}) {
  const [selected, setSelected] = useState<string[]>([...initial]);
  return (
    <Combobox items={[...groups]} multiple value={selected} onValueChange={setSelected} defaultOpen={open}>
      <ComboboxTrigger render={<InputGroupButton size="xs" aria-label={label} />}>
        {icon}
        {label}
        {selected.length > 0 && (
          <Badge variant="secondary" className="h-4 min-w-4 px-1 tabular-nums">
            {selected.length}
          </Badge>
        )}
      </ComboboxTrigger>
      <ComboboxContent side="top" className="w-72">
        <ComboboxInput showTrigger={false} placeholder={`Search ${label.toLowerCase()}`} />
        <ComboboxEmpty>Nothing matches</ComboboxEmpty>
        <GroupedList />
      </ComboboxContent>
    </Combobox>
  );
}
