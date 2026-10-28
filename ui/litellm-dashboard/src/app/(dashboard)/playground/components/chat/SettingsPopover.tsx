"use client";

import { useState } from "react";
import { ChevronRight, Minus, Plus, SlidersHorizontal } from "lucide-react";
import type { KeyValuePair } from "@/components/key_value_input";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldSet } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { InputGroupButton } from "@/components/ui/input-group";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { Capabilities } from "./endpointCapabilities";
import { CopyIconButton } from "./ModelName";
import type { KeySource, PlaygroundSettings } from "./playgroundSettings";
import type { SettingsUpdate } from "./usePlaygroundSettings";

const KEY_SOURCES = [
  { value: "session", label: "Current UI session" },
  { value: "custom", label: "Virtual key" },
];

const DEFAULT_TEMPERATURE = 1;

function TagField({
  value,
  options,
  onChange,
}: {
  value: readonly string[];
  options: readonly string[];
  onChange: (value: readonly string[]) => void;
}) {
  const anchor = useComboboxAnchor();
  const [query, setQuery] = useState("");
  const custom = query.trim();
  const known = [...new Set([...options, ...value])];
  const items = custom === "" || known.includes(custom) ? known : [...known, custom];
  return (
    <Combobox
      items={items}
      multiple
      value={[...value]}
      onValueChange={(next: string[]) => onChange(next)}
      inputValue={query}
      onInputValueChange={setQuery}
      autoHighlight
    >
      <ComboboxChips ref={anchor} className="min-h-8 py-1">
        <ComboboxValue>
          {(values: string[]) => values.map((tag) => <ComboboxChip key={tag}>{tag}</ComboboxChip>)}
        </ComboboxValue>
        <ComboboxChipsInput id="playground-tags" placeholder={value.length === 0 ? "Add a tag" : ""} />
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>No tags yet, type one to add it</ComboboxEmpty>
        <ComboboxList>
          {(tag: string) => (
            <ComboboxItem key={tag} value={tag}>
              {known.includes(tag) ? (
                tag
              ) : (
                <>
                  <Plus />
                  Add {tag}
                </>
              )}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}

function HeaderEditor({
  pairs,
  onChange,
}: {
  pairs: readonly KeyValuePair[];
  onChange: (pairs: readonly KeyValuePair[]) => void;
}) {
  const replace = (index: number, pair: KeyValuePair) => onChange(pairs.map((item, i) => (i === index ? pair : item)));
  return (
    <div className="flex flex-col gap-2">
      {pairs.map(([name, value], index) => (
        <ButtonGroup key={index} className="w-full">
          <Input
            aria-label="Header name"
            value={name}
            onChange={(event) => replace(index, [event.target.value, value])}
            placeholder="Name"
            className="h-8 font-mono text-xs"
          />
          <Input
            aria-label="Header value"
            value={value}
            onChange={(event) => replace(index, [name, event.target.value])}
            placeholder="Value"
            className="h-8 font-mono text-xs"
          />
          <Button
            variant="outline"
            size="icon-sm"
            aria-label={`Remove ${name || "header"}`}
            onClick={() => onChange(pairs.filter((_, i) => i !== index))}
          >
            <Minus />
          </Button>
        </ButtonGroup>
      ))}
      <Button variant="ghost" size="sm" className="-ml-2 w-fit" onClick={() => onChange([...pairs, ["", ""]])}>
        <Plus />
        Add header
      </Button>
    </div>
  );
}

function ProxyBaseUrlField({
  value,
  suggested,
  onChange,
}: {
  value: string;
  suggested: string | null;
  onChange: (value: string) => void;
}) {
  return (
    <Field>
      <div className="flex items-center justify-between">
        <FieldLabel htmlFor="playground-base-url">Proxy base URL</FieldLabel>
        {value ? (
          <Button variant="link" size="xs" className="h-auto p-0" onClick={() => onChange("")}>
            Clear
          </Button>
        ) : (
          suggested && (
            <Button variant="link" size="xs" className="h-auto p-0" onClick={() => onChange(suggested)}>
              Use {suggested}
            </Button>
          )
        )}
      </div>
      <Input
        id="playground-base-url"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Same as this dashboard"
        className="h-8"
      />
      {value && <FieldDescription>Requests go to {value}</FieldDescription>}
    </Field>
  );
}

const parseTemperature = (text: string): number | null | undefined => {
  if (text.trim() === "") return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 2 ? parsed : undefined;
};

function TemperatureInput({ value, onChange }: { value: number | null; onChange: (value: number | null) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const edit = (text: string) => {
    setDraft(text);
    const parsed = parseTemperature(text);
    if (parsed !== undefined) onChange(parsed);
  };
  return (
    <Input
      aria-label="Temperature value"
      inputMode="decimal"
      placeholder="Default"
      value={draft ?? value?.toString() ?? ""}
      onChange={(event) => edit(event.target.value)}
      onBlur={() => setDraft(null)}
      className="h-8 w-16 shrink-0 text-right text-xs tabular-nums"
    />
  );
}

function SamplingFields({
  settings,
  update,
}: {
  settings: PlaygroundSettings;
  update: (change: SettingsUpdate) => void;
}) {
  const temperature = settings.temperature ?? DEFAULT_TEMPERATURE;
  return (
    <>
      <Field orientation="horizontal">
        <FieldLabel htmlFor="playground-temperature" className="w-28 shrink-0 font-normal">
          Temperature
        </FieldLabel>
        <Slider
          id="playground-temperature"
          min={0}
          max={2}
          step={0.1}
          value={[temperature]}
          onValueChange={(value) => update({ temperature: Array.isArray(value) ? value[0] : value })}
        />
        <TemperatureInput value={settings.temperature} onChange={(temperature) => update({ temperature })} />
      </Field>
      <Field orientation="horizontal">
        <FieldLabel htmlFor="playground-max-tokens" className="w-28 shrink-0 font-normal">
          Max tokens
        </FieldLabel>
        <Input
          id="playground-max-tokens"
          type="number"
          min={1}
          value={settings.maxTokens ?? ""}
          onChange={(event) => {
            const parsed = Number.parseInt(event.target.value, 10);
            update({ maxTokens: Number.isFinite(parsed) && parsed > 0 ? parsed : null });
          }}
          placeholder="Default"
          className="h-8 w-24"
        />
      </Field>
    </>
  );
}

export function SettingsPopover({
  settings,
  update,
  capabilities,
  keysDisabled,
  tagOptions,
  suggestedBaseUrl,
  sessions,
  defaultOpen = false,
}: {
  settings: PlaygroundSettings;
  update: (change: SettingsUpdate) => void;
  capabilities: Capabilities;
  keysDisabled: boolean;
  tagOptions: readonly string[];
  suggestedBaseUrl: string | null;
  sessions: Readonly<{ enabled: boolean; responseId: string | null; onToggle: (enabled: boolean) => void }>;
  defaultOpen?: boolean;
}) {
  const advancedSet = [
    settings.headerPairs.length > 0,
    Boolean(settings.proxyBaseUrl),
    settings.temperature !== null,
    settings.maxTokens !== null,
  ].some(Boolean);
  const [advancedOpen, setAdvancedOpen] = useState(advancedSet);
  const hasToggles = capabilities.stream || capabilities.simulateFallback || capabilities.sessions;
  const lastResponseId = capabilities.sessions && sessions.enabled ? sessions.responseId : null;
  return (
    <Popover defaultOpen={defaultOpen}>
      <PopoverTrigger render={<InputGroupButton size="xs" />}>
        <SlidersHorizontal />
        Settings
      </PopoverTrigger>
      <PopoverContent align="end" side="top" className="max-h-(--available-height) w-96 overflow-y-auto p-0">
        <FieldGroup className="gap-5 p-4">
          <Field>
            <FieldLabel htmlFor="playground-key-source">Send as</FieldLabel>
            <Select
              items={KEY_SOURCES}
              value={settings.keySource}
              onValueChange={(value) => update({ keySource: value as KeySource })}
              disabled={keysDisabled}
            >
              <SelectTrigger id="playground-key-source" size="sm" className="w-full" aria-label="Send as">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {KEY_SOURCES.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {settings.keySource === "custom" ? (
              <PasswordInput
                aria-label="Virtual key"
                value={settings.virtualKey}
                onChange={(event) => update({ virtualKey: event.target.value })}
                placeholder="Paste a virtual key"
              />
            ) : (
              <FieldDescription>Pick a virtual key to test what it can access</FieldDescription>
            )}
          </Field>
          <Field>
            <FieldLabel htmlFor="playground-tags">Tags</FieldLabel>
            <TagField value={settings.tags} options={tagOptions} onChange={(tags) => update({ tags })} />
          </Field>
          {hasToggles && (
            <FieldSet className="gap-3">
              {capabilities.stream && (
                <Field orientation="horizontal">
                  <FieldLabel htmlFor="playground-stream" className="font-normal">
                    Stream response
                  </FieldLabel>
                  <Switch
                    id="playground-stream"
                    checked={settings.stream}
                    onCheckedChange={(stream) => update({ stream })}
                  />
                </Field>
              )}
              {capabilities.simulateFallback && (
                <Field orientation="horizontal">
                  <FieldLabel htmlFor="playground-fallback" className="font-normal">
                    Simulate a fallback
                  </FieldLabel>
                  <Switch
                    id="playground-fallback"
                    checked={settings.simulateFallback}
                    onCheckedChange={(simulateFallback) => update({ simulateFallback })}
                  />
                </Field>
              )}
              {capabilities.sessions && (
                <Field orientation="horizontal">
                  <FieldLabel htmlFor="playground-sessions" className="flex-col items-start gap-0.5 font-normal">
                    Keep context on the gateway
                    <span className="text-xs text-muted-foreground">
                      Sends previous_response_id instead of the whole history
                    </span>
                  </FieldLabel>
                  <Switch id="playground-sessions" checked={sessions.enabled} onCheckedChange={sessions.onToggle} />
                </Field>
              )}
              {lastResponseId && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="shrink-0">Last response</span>
                  <span className="min-w-0 truncate font-mono">{lastResponseId}</span>
                  <CopyIconButton value={lastResponseId} label="response ID" />
                </div>
              )}
            </FieldSet>
          )}
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger
              render={
                <Button
                  variant="ghost"
                  size="sm"
                  className="-ml-2 w-fit text-muted-foreground aria-expanded:bg-transparent"
                />
              }
            >
              <ChevronRight className="transition-transform in-data-panel-open:rotate-90" />
              Advanced
            </CollapsibleTrigger>
            <CollapsibleContent>
              <FieldGroup className="gap-5 pt-3">
                {capabilities.customHeaders && (
                  <Field>
                    <FieldLabel>Custom headers</FieldLabel>
                    <HeaderEditor pairs={settings.headerPairs} onChange={(headerPairs) => update({ headerPairs })} />
                    <FieldDescription>Sent with every request, like anthropic-beta</FieldDescription>
                  </Field>
                )}
                <ProxyBaseUrlField
                  value={settings.proxyBaseUrl}
                  suggested={suggestedBaseUrl}
                  onChange={(proxyBaseUrl) => update({ proxyBaseUrl })}
                />
                {capabilities.sampling && <SamplingFields settings={settings} update={update} />}
              </FieldGroup>
            </CollapsibleContent>
          </Collapsible>
        </FieldGroup>
      </PopoverContent>
    </Popover>
  );
}
