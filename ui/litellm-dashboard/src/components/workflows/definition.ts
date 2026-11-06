import { z } from "zod";

const jsonSchema = z.record(z.string(), z.unknown());
const reference = z.object({ from: z.string().min(1), output: z.string().min(1) });
const binding = z.union([reference, z.object({ value: z.json() })]);
const commonNode = {
  id: z.string().min(1),
  timeout_seconds: z.number().positive(),
  retry: z.object({ max_attempts: z.number().int().positive(), backoff_seconds: z.number().nonnegative() }),
};
const structuredOutputFields = { name: z.string(), strict: z.boolean(), schema: jsonSchema };
const chatNodeFields = {
  ...commonNode,
  type: z.literal("chat_completion"),
  model: z.string().min(1),
  messages: binding,
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  max_tokens: z.number().optional(),
  max_completion_tokens: z.number().optional(),
  seed: z.number().optional(),
  stop: z.union([z.string(), z.array(z.string())]).optional(),
  response_format: z
    .object({
      type: z.literal("json_schema"),
      json_schema: z.object(structuredOutputFields),
    })
    .optional(),
};
const chatNode = z.looseObject(chatNodeFields);
const programFields = {
  contract_version: z.literal("1"),
  source: z.string(),
  function_name: z.string().min(1),
  parameter_names: z.array(z.string()),
  input_schema: jsonSchema,
  output_schema: jsonSchema,
  source_sha256: z.string(),
};
const pythonNodeFields = {
  ...commonNode,
  type: z.literal("python"),
  inputs: z.record(z.string(), binding),
  program: z.object(programFields),
};
const pythonNode = z.looseObject(pythonNodeFields);

const compiledBundleFields = {
  contract_version: z.literal("1"),
  name: z.string().min(1),
  timeout_seconds: z.number().positive(),
  input_schema: jsonSchema,
  output: reference,
  nodes: z.array(z.discriminatedUnion("type", [chatNode, pythonNode])),
};
const compiledBundle = z.looseObject(compiledBundleFields);

export type CompiledWorkflowBundle = z.infer<typeof compiledBundle>;
export type CompiledWorkflowNode = CompiledWorkflowBundle["nodes"][number];
export type WorkflowBinding = z.infer<typeof binding>;
export type WorkflowNodeState = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface WorkflowExecutionOverlay {
  definitionKey: string;
  states: Readonly<Record<string, WorkflowNodeState>>;
}

export type WorkflowDefinition =
  | { kind: "ready"; bundle: CompiledWorkflowBundle }
  | { kind: "empty" }
  | { kind: "unsupported"; detail: string }
  | { kind: "invalid" };

const envelope = z.object({
  contract_version: z.string(),
  nodes: z.array(z.looseObject({ type: z.string() })),
});

export function readWorkflowDefinition(value: unknown): WorkflowDefinition {
  if (value == null) return { kind: "empty" };
  const outer = envelope.safeParse(value);
  if (!outer.success) return { kind: "invalid" };
  if (outer.data.contract_version !== "1") {
    return { kind: "unsupported", detail: `contract_version ${outer.data.contract_version}` };
  }
  const unsupported = outer.data.nodes.filter((node) => !["chat_completion", "python"].includes(node.type));
  if (unsupported.length > 0) {
    return { kind: "unsupported", detail: [...new Set(unsupported.map((node) => node.type))].join(", ") };
  }
  const parsed = compiledBundle.safeParse(value);
  if (!parsed.success) return { kind: "invalid" };
  const ids = parsed.data.nodes.map((node) => node.id);
  if (new Set(ids).size !== ids.length) return { kind: "invalid" };
  if (ids.length === 0) return { kind: "empty" };
  return { kind: "ready", bundle: parsed.data };
}

export function nodeTitle(node: CompiledWorkflowNode): string {
  return node.type === "chat_completion" ? node.model : node.program.function_name;
}

export function sequentialConnections(nodes: readonly CompiledWorkflowNode[]) {
  return nodes.slice(1).map((node, index) => ({ source: nodes[index].id, target: node.id }));
}

export function executionStates(definitionKey: string, overlay?: WorkflowExecutionOverlay) {
  return overlay?.definitionKey === definitionKey ? overlay.states : undefined;
}
