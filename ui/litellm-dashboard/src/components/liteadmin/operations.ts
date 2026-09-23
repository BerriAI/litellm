import { zodFunction } from "openai/helpers/zod";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { ApiError } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";
import { generatedKey, projectToolResult, type ResultKind } from "./toolResults";

export interface LiteAdminAction {
  id: string;
  name: string;
  title: string;
  arguments: Record<string, unknown>;
  destructive: boolean;
}

export type ActionResult = { status: "completed"; key?: string } | { status: "unknown"; message: string };

export interface OperationContext {
  accessToken: string;
  signal: AbortSignal;
  assertCurrent: () => void;
  beforeTool: () => void;
  confirm: (action: LiteAdminAction) => Promise<boolean>;
  onResult: (action: LiteAdminAction, result: ActionResult) => void;
}

type Schemas = components["schemas"];
const text = z.string().min(1).max(200);
const hash = z.string().regex(/^[a-f0-9]{64}$/i, "Use the key hash from a lookup, not a raw API key.");
const optional = <Schema extends z.ZodType<unknown>>(schema: Schema) =>
  schema.nullable().transform((value) => value ?? undefined);
const optionalText = optional(text);
const amount = optional(z.number().finite().nonnegative());
const limit = optional(z.number().int().nonnegative());
const models = optional(z.array(text).max(50));
const page = z.number().int().min(1);
const pageSize = z.number().int().min(1).max(50);
const limits = { max_budget: amount, budget_duration: optionalText, rpm_limit: limit, tpm_limit: limit };
const keyFields = {
  key_alias: optionalText,
  team_id: optionalText,
  user_id: optionalText,
  models,
  ...limits,
  budget_id: optionalText,
  duration: optionalText,
};
const teamFields = { team_alias: optionalText, organization_id: optionalText, models, ...limits };
const userFields = {
  user_email: optional(z.string().email().max(200)),
  user_alias: optionalText,
  user_role: optional(z.enum(["proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer"])),
  models,
  ...limits,
};
const object = <Shape extends z.ZodRawShape>(shape: Shape) => z.object(shape).strict();
const dated = <Shape extends z.ZodRawShape>(shape: Shape) =>
  object({ start_date: z.string().date(), end_date: z.string().date(), ...shape }).refine((value) => {
    if (typeof value.start_date !== "string" || typeof value.end_date !== "string") return false;
    const days = (Date.parse(value.end_date) - Date.parse(value.start_date)) / 86_400_000;
    return days >= 0 && days <= 366;
  }, "Choose an ordered date range of at most 366 days.");

const keysListFields = {
  page,
  page_size: pageSize,
  search: optionalText,
  team_id: optionalText,
  user_id: optionalText,
};
const teamsListFields = { page, page_size: pageSize, search: optionalText, organization_id: optionalText };
const teamMemberAddFields = {
  team_id: text,
  member: object({ user_id: text, role: z.enum(["admin", "user"]) }),
  max_budget_in_team: amount,
  budget_duration: optionalText,
};
const teamMemberUpdateFields = {
  team_id: text,
  user_id: text,
  role: optional(z.enum(["admin", "user"])),
  max_budget_in_team: amount,
  budget_duration: optionalText,
  rpm_limit: limit,
  tpm_limit: limit,
};
const spendReportFields = {
  group_by: z.enum(["team", "customer", "api_key"]),
  api_key: optional(hash),
  team_id: optionalText,
  internal_user_id: optionalText,
  customer_id: optionalText,
};
const requestLogsFields = {
  page,
  page_size: pageSize,
  request_id: optionalText,
  team_id: optionalText,
  user_id: optionalText,
  model: optionalText,
  status_filter: optional(z.enum(["success", "failure"])),
};

export function createLiteAdminOperations(context: OperationContext) {
  const active = () => {
    context.signal.throwIfAborted();
    context.assertCurrent();
  };
  const auth = { accessToken: context.accessToken, signal: context.signal };
  const operation =
    (mode: "read" | "write" | "delete", kind: ResultKind) =>
    <Schema extends z.ZodType<Record<string, unknown>>>(
      name: string,
      title: string,
      schema: Schema,
      execute: (args: z.output<Schema>) => Promise<unknown>,
    ) => {
      const definition = {
        name,
        description: `${title}. Null means omitted or unchanged. ${mode === "read" ? "Read-only." : "Requires the administrator to review and confirm the change."}`,
        parameters: schema,
        function: async (args: z.output<Schema>) => {
          active();
          context.beforeTool();
          const action: LiteAdminAction | undefined =
            mode === "read"
              ? undefined
              : {
                  id: crypto.randomUUID(),
                  name,
                  title,
                  arguments: Object.fromEntries(Object.entries(args).filter(([, value]) => value !== undefined)),
                  destructive: mode === "delete",
                };
          if (action) {
            const approved = await context.confirm(action);
            active();
            if (!approved) throw new Error("Action cancelled. No change was sent.");
          }
          const outcome = await Promise.resolve()
            .then(() => {
              active();
              return execute(args);
            })
            .then(
              (value) => ({ ok: true, value }) as const,
              (error: unknown) => ({ ok: false, error }) as const,
            );
          active();
          if (!outcome.ok) {
            if (action) {
              const result: ActionResult = {
                status: "unknown",
                message: "The change could not be verified. Check the resource before trying again.",
              };
              context.onResult(action, result);
              throw new Error(result.message);
            }
            return {
              operation: name,
              success: false,
              status: outcome.error instanceof ApiError ? outcome.error.status : undefined,
              message: "The gateway could not complete this lookup.",
            };
          }
          const key = name === "key_create" || name === "user_create" ? generatedKey(outcome.value) : undefined;
          if (action) context.onResult(action, { status: "completed", ...(key ? { key } : {}) });
          return {
            operation: name,
            success: true,
            result: projectToolResult(kind, outcome.value, [context.accessToken, ...(key ? [key] : [])]),
          };
        },
      };
      return zodFunction(definition);
    };

  return [
    operation("read", "key")("keys_list", "List virtual keys", object(keysListFields), (a) =>
      apiClient.get<unknown>("/key/list", {
        ...auth,
        query: {
          page: a.page,
          size: a.page_size,
          search: a.search,
          team_id: a.team_id,
          user_id: a.user_id,
          return_full_object: true,
        },
      }),
    ),
    operation("read", "key")("key_info", "View a virtual key", object({ key: hash }), (a) =>
      apiClient.get<unknown>("/key/info", { ...auth, query: a }),
    ),
    operation("write", "key")("key_create", "Create a virtual key", object(keyFields), (a) =>
      apiClient.post<unknown>("/key/generate", { ...auth, body: a satisfies Partial<Schemas["GenerateKeyRequest"]> }),
    ),
    operation("write", "key")("key_update", "Update a virtual key", object({ key: hash, ...keyFields }), (a) =>
      apiClient.post<unknown>("/key/update", { ...auth, body: a satisfies Partial<Schemas["UpdateKeyRequest"]> }),
    ),
    operation("delete", "key")(
      "key_delete",
      "Delete virtual keys",
      object({ keys: z.array(hash).min(1).max(20) }),
      (a) => apiClient.post<unknown>("/key/delete", { ...auth, body: a satisfies Schemas["KeyRequest"] }),
    ),
    operation("write", "key")("key_block", "Block a virtual key", object({ key: hash }), (a) =>
      apiClient.post<unknown>("/key/block", { ...auth, body: a satisfies Schemas["BlockKeyRequest"] }),
    ),
    operation("write", "key")("key_unblock", "Unblock a virtual key", object({ key: hash }), (a) =>
      apiClient.post<unknown>("/key/unblock", { ...auth, body: a satisfies Schemas["BlockKeyRequest"] }),
    ),
    operation("read", "team")("teams_list", "List teams", object(teamsListFields), (a) =>
      apiClient.get<unknown>("/v2/team/list", { ...auth, query: a }),
    ),
    operation("read", "team")("team_info", "View a team and its members", object({ team_id: text }), (a) =>
      apiClient.get<unknown>("/team/info", { ...auth, query: { ...a, key_limit: 20 } }),
    ),
    operation("write", "team")("team_create", "Create a team", object(teamFields), (a) =>
      apiClient.post<unknown>("/team/new", { ...auth, body: a satisfies Partial<Schemas["NewTeamRequest"]> }),
    ),
    operation("write", "team")("team_update", "Update a team", object({ team_id: text, ...teamFields }), (a) =>
      apiClient.post<unknown>("/team/update", { ...auth, body: a satisfies Schemas["UpdateTeamRequest"] }),
    ),
    operation("delete", "team")(
      "team_delete",
      "Delete teams",
      object({ team_ids: z.array(text).min(1).max(20) }),
      (a) => apiClient.post<unknown>("/team/delete", { ...auth, body: a satisfies Schemas["DeleteTeamRequest"] }),
    ),
    operation("write", "team")("team_member_add", "Add a team member", object(teamMemberAddFields), (a) =>
      apiClient.post<unknown>("/team/member_add", { ...auth, body: a satisfies Schemas["TeamMemberAddRequest"] }),
    ),
    operation("write", "team")("team_member_update", "Update a team member", object(teamMemberUpdateFields), (a) =>
      apiClient.post<unknown>("/team/member_update", {
        ...auth,
        body: a satisfies Schemas["TeamMemberUpdateRequest"],
      }),
    ),
    operation("delete", "team")(
      "team_member_delete",
      "Remove a team member",
      object({ team_id: text, user_id: text }),
      (a) =>
        apiClient.post<unknown>("/team/member_delete", {
          ...auth,
          body: a satisfies Schemas["TeamMemberDeleteRequest"],
        }),
    ),
    operation("read", "user")(
      "users_list",
      "List users",
      object({ page, page_size: pageSize, search: optionalText }),
      (a) => apiClient.get<unknown>("/user/list", { ...auth, query: a }),
    ),
    operation("read", "user")("user_info", "View a user", object({ user_id: text }), (a) =>
      apiClient.get<unknown>("/v2/user/info", { ...auth, query: a }),
    ),
    operation("write", "user")(
      "user_create",
      "Create a user without a key",
      object({ user_id: optionalText, ...userFields }).transform((a) => ({ ...a, auto_create_key: false })),
      (a) => apiClient.post<unknown>("/user/new", { ...auth, body: a satisfies Partial<Schemas["NewUserRequest"]> }),
    ),
    operation("write", "user")("user_update", "Update a user", object({ user_id: text, ...userFields }), (a) =>
      apiClient.post<unknown>("/user/update", { ...auth, body: a satisfies Partial<Schemas["UpdateUserRequest"]> }),
    ),
    operation("delete", "user")(
      "user_delete",
      "Delete users",
      object({ user_ids: z.array(text).min(1).max(20) }),
      (a) => apiClient.post<unknown>("/user/delete", { ...auth, body: a satisfies Schemas["DeleteUserRequest"] }),
    ),
    operation("read", "budget")(
      "budgets_list",
      "List budgets",
      object({ page, page_size: pageSize, search: optionalText }),
      (a) =>
        apiClient.get<unknown>("/management/v1/budgets", {
          ...auth,
          query: { page: a.page, page_size: a.page_size, q: a.search },
        }),
    ),
    operation("read", "budget")("budget_info", "View budgets", object({ budgets: z.array(text).min(1).max(20) }), (a) =>
      apiClient.post<unknown>("/budget/info", { ...auth, body: a satisfies Schemas["BudgetRequest"] }),
    ),
    operation("write", "budget")(
      "budget_create",
      "Create a budget",
      object({ budget_id: optionalText, ...limits }),
      (a) => apiClient.post<unknown>("/budget/new", { ...auth, body: a satisfies Schemas["BudgetNewRequest"] }),
    ),
    operation("write", "budget")("budget_update", "Update a budget", object({ budget_id: text, ...limits }), (a) =>
      apiClient.post<unknown>("/budget/update", { ...auth, body: a satisfies Schemas["BudgetNewRequest"] }),
    ),
    operation("delete", "budget")("budget_delete", "Delete a budget", object({ id: text }), (a) =>
      apiClient.post<unknown>("/budget/delete", { ...auth, body: a satisfies Schemas["BudgetDeleteRequest"] }),
    ),
    operation("read", "spend")(
      "spend_report",
      "View spend by date and group (requires an enterprise license)",
      dated(spendReportFields),
      (a) => apiClient.get<unknown>("/global/spend/report", { ...auth, query: a }),
    ),
    operation("read", "spend")(
      "team_spend_report",
      "View a team's spend by model and key (requires an enterprise license)",
      dated({ team_id: text }),
      (a) => apiClient.get<unknown>("/team/spend/report", { ...auth, query: a }),
    ),
    operation("read", "spend")(
      "key_spend_report",
      "View a key's spend by model (requires an enterprise license)",
      dated({ api_key: hash }),
      (a) => apiClient.get<unknown>("/key/spend/report", { ...auth, query: a }),
    ),
    operation("read", "log")(
      "request_logs",
      "List request cost, timing and status, excluding prompts and responses",
      dated(requestLogsFields),
      (a) =>
        apiClient.get<unknown>("/spend/logs/ui", {
          ...auth,
          query: { ...a, start_date: `${a.start_date} 00:00:00`, end_date: `${a.end_date} 23:59:59` },
        }),
    ),
  ];
}
