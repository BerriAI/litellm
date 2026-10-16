"use client";

import { useState } from "react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Plus } from "lucide-react";
import { ENDPOINT_OPTIONS } from "../chat_ui/chatConstants";
import { PROTOTYPE_MODEL } from "./PrototypeThread";

const MODEL_OPTIONS = [PROTOTYPE_MODEL, "gpt-5.5", "anthropic-sonnet-5", "anthropic-opus-4-8", "vertex-sonnet-4-6"].map(
  (model) => ({ value: model, label: model }),
);
const GUARDRAIL_OPTIONS = ["pii-mask", "prompt-injection", "toxicity"].map((g) => ({ value: g, label: g }));
const MCP_OPTIONS = ["github", "linear", "slack"].map((s) => ({ value: s, label: s }));
const VECTOR_STORE_OPTIONS = ["support-docs", "eng-wiki"].map((s) => ({ value: s, label: s }));
const KEY_SOURCE_ITEMS = [
  { value: "session", label: "Current UI session" },
  { value: "custom", label: "Virtual key" },
];

export function EndpointField({ label = true }: { label?: boolean }) {
  const [endpoint, setEndpoint] = useState<string | null>("chat");
  return (
    <Field>
      {label && <FieldLabel htmlFor="proto-endpoint">Endpoint</FieldLabel>}
      <SearchSelect
        id="proto-endpoint"
        aria-label="Endpoint"
        options={ENDPOINT_OPTIONS}
        value={endpoint}
        onValueChange={setEndpoint}
      />
    </Field>
  );
}

export function ModelField({ label = true }: { label?: boolean }) {
  const [model, setModel] = useState<string | null>(PROTOTYPE_MODEL);
  return (
    <Field>
      {label && <FieldLabel htmlFor="proto-model">Model</FieldLabel>}
      <SearchSelect
        id="proto-model"
        aria-label="Model"
        options={MODEL_OPTIONS}
        value={model}
        onValueChange={setModel}
        placeholder="Select a model"
      />
    </Field>
  );
}

export function ParameterFields() {
  return (
    <FieldGroup className="gap-4">
      <div className="grid grid-cols-2 gap-2">
        <Field>
          <FieldLabel htmlFor="proto-temp">Temperature</FieldLabel>
          <Input id="proto-temp" defaultValue="1.0" className="h-8" />
        </Field>
        <Field>
          <FieldLabel htmlFor="proto-max">Max tokens</FieldLabel>
          <Input id="proto-max" defaultValue="2048" className="h-8" />
        </Field>
      </div>
      <Field orientation="horizontal">
        <FieldLabel htmlFor="proto-stream">Stream response</FieldLabel>
        <Switch id="proto-stream" defaultChecked />
      </Field>
      <Field orientation="horizontal">
        <FieldLabel htmlFor="proto-fallback">Simulate a fallback</FieldLabel>
        <Switch id="proto-fallback" />
      </Field>
    </FieldGroup>
  );
}

export function ToolFields() {
  const [mcp, setMcp] = useState<string[]>(["github"]);
  const [stores, setStores] = useState<string[]>([]);
  return (
    <FieldGroup className="gap-4">
      <Field>
        <FieldLabel htmlFor="proto-mcp">MCP servers</FieldLabel>
        <MultiSelect id="proto-mcp" options={MCP_OPTIONS} value={mcp} onValueChange={setMcp} placeholder="None" />
      </Field>
      <Field>
        <FieldLabel htmlFor="proto-vs">Vector stores</FieldLabel>
        <MultiSelect
          id="proto-vs"
          options={VECTOR_STORE_OPTIONS}
          value={stores}
          onValueChange={setStores}
          placeholder="None"
        />
      </Field>
    </FieldGroup>
  );
}

export function SafetyFields() {
  const [guardrails, setGuardrails] = useState<string[]>(["pii-mask"]);
  const [policies, setPolicies] = useState<string[]>([]);
  return (
    <FieldGroup className="gap-4">
      <Field>
        <FieldLabel htmlFor="proto-guardrails">Guardrails</FieldLabel>
        <MultiSelect
          id="proto-guardrails"
          options={GUARDRAIL_OPTIONS}
          value={guardrails}
          onValueChange={setGuardrails}
          placeholder="None"
        />
      </Field>
      <Field>
        <FieldLabel htmlFor="proto-policies">Policies</FieldLabel>
        <MultiSelect id="proto-policies" options={[]} value={policies} onValueChange={setPolicies} placeholder="None" />
      </Field>
    </FieldGroup>
  );
}

export function RequestFields() {
  const [source, setSource] = useState("session");
  return (
    <FieldGroup className="gap-4">
      <Field>
        <FieldLabel htmlFor="proto-key">Key</FieldLabel>
        <Select items={KEY_SOURCE_ITEMS} value={source} onValueChange={(v) => setSource(v as string)}>
          <SelectTrigger id="proto-key" size="sm" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {KEY_SOURCE_ITEMS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <Field>
        <FieldLabel>Custom headers</FieldLabel>
        <Button variant="outline" size="sm" className="w-fit!">
          <Plus />
          Add header
        </Button>
        <FieldDescription>Sent with every request, for example anthropic-beta</FieldDescription>
      </Field>
      <Field>
        <FieldLabel htmlFor="proto-proxy">Proxy base URL</FieldLabel>
        <Input id="proto-proxy" placeholder="http://localhost:4000" className="h-8" />
      </Field>
    </FieldGroup>
  );
}
