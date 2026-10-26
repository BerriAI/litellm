import { z } from "zod";

export type ResultKind = "key" | "team" | "user" | "budget" | "spend" | "usage" | "log";

const limitedList = <Schema extends z.ZodType<unknown>>(schema: Schema, limit = 50) =>
  z.preprocess((value) => (Array.isArray(value) ? value.slice(0, limit) : value), z.array(schema));

function sumKnownTokenCounts(left: number | null | undefined, right: number | null | undefined): number | undefined {
  return left == null || right == null ? undefined : left + right;
}

export function generatedKey(value: unknown): string | undefined {
  const result = z.object({ key: z.string().min(1) }).safeParse(value);
  return result.success ? result.data.key : undefined;
}

export function projectToolResult(kind: ResultKind, value: unknown, secrets: readonly string[]): unknown {
  const text = z
    .string()
    .transform((value) =>
      secrets
        .filter(Boolean)
        .reduce((safe, secret) => safe.split(secret).join("[redacted]"), value)
        .replace(/\bsk-[\w-]+/g, "[redacted]")
        .replace(/\bBearer\s+\S+/gi, "[redacted]")
        .slice(0, 400),
    )
    .nullish();
  const number = z.number().finite().nullish();
  const hash = z
    .string()
    .regex(/^[a-f0-9]{64}$/i)
    .transform((value) => (secrets.includes(value) ? undefined : value))
    .nullish()
    .catch(undefined);
  const policy = {
    spend: number,
    max_budget: number,
    soft_budget: number,
    budget_id: text,
    budget_duration: text,
    budget_reset_at: text,
    rpm_limit: number,
    tpm_limit: number,
    blocked: z.boolean().nullish(),
    models: limitedList(text).nullish(),
  };
  const pagination = {
    page: number,
    current_page: number,
    page_size: number,
    total: number,
    total_count: number,
    total_pages: number,
  };
  const keyFields = {
    ...policy,
    token: hash,
    key_hash: hash,
    key_alias: text,
    key_name: text,
    team_id: text,
    user_id: text,
    organization_id: text,
    expires: text,
  };
  const key = z.object(keyFields);
  const memberFields = {
    user_id: text,
    user_email: text,
    role: text,
    spend: number,
    max_budget_in_team: number,
  };
  const member = z.object(memberFields);
  const teamFields = {
    ...policy,
    team_id: text,
    team_alias: text,
    organization_id: text,
    members_with_roles: limitedList(member).nullish(),
  };
  const team = z.object(teamFields);
  const userFields = {
    ...policy,
    user_id: text,
    user_alias: text,
    user_email: text,
    user_role: text,
    teams: limitedList(text).nullish(),
  };
  const user = z.object(userFields);
  const budget = z.object({ ...policy, created_at: text, updated_at: text });
  const costsFields = {
    model: text,
    api_key: hash,
    spend: number,
    total_spend: number,
    total_cost: number,
    total_tokens: number,
    total_input_tokens: number,
    total_output_tokens: number,
  };
  const costs = z.object(costsFields);
  const spendGroupFields = {
    team_id: text,
    team_alias: text,
    team_name: text,
    customer: text,
    metadata: limitedList(costs, 10).nullish(),
    model_details: limitedList(costs, 10).nullish(),
  };
  const spendGroup = costs.extend(spendGroupFields);
  const spendFields = {
    group_by_day: text,
    date: text,
    teams: limitedList(spendGroup, 10).nullish(),
    customers: limitedList(spendGroup, 10).nullish(),
  };
  const spend = spendGroup.extend(spendFields);
  const logFields = {
    request_id: text,
    model: text,
    model_id: text,
    team_id: text,
    user: text,
    call_type: text,
    status: text,
    spend: number,
    total_tokens: number,
    prompt_tokens: number,
    completion_tokens: number,
    startTime: text,
    endTime: text,
    request_duration_ms: number,
    ttft_ms: number,
    cache_hit: z.union([text, z.boolean()]),
  };
  const log = z.object(logFields);
  const usageMetricFields = {
    spend: z.number().finite(),
    api_requests: z.number().finite(),
    total_tokens: z.number().finite(),
    prompt_tokens: number,
    completion_tokens: number,
    cache_read_input_tokens: number,
    cache_creation_input_tokens: number,
  };
  const usageMetrics = z.object(usageMetricFields);
  const usageEntry = z.object({
    metrics: usageMetrics,
    metadata: z.object({ key_alias: text, team_alias: text }).optional(),
  });
  const usageBreakdown = z.record(z.string(), usageEntry).optional();
  const usageBreakdownFields = {
    models: usageBreakdown,
    providers: usageBreakdown,
    entities: usageBreakdown,
    api_keys: usageBreakdown,
  };
  const usageMetadataFields = {
    total_spend: number,
    total_api_requests: number,
    total_tokens: number,
    total_prompt_tokens: number,
    total_completion_tokens: number,
    total_cache_read_input_tokens: number,
    total_cache_creation_input_tokens: number,
    total_successful_requests: number,
    total_failed_requests: number,
    total_api_keys: number,
    api_key_limit: number,
  };
  const usage = z
    .object({
      results: z
        .array(
          z.object({
            date: text,
            metrics: usageMetrics,
            breakdown: z.object(usageBreakdownFields).optional(),
          }),
        )
        .max(367),
      metadata: z.object(usageMetadataFields).optional(),
    })
    .transform(({ results, metadata }) => {
      const top = (dimension: "models" | "providers" | "entities" | "api_keys") => {
        const totals = new Map<string, z.infer<typeof usageEntry>>();
        for (const day of results) {
          for (const [name, entry] of Object.entries(day.breakdown?.[dimension] ?? {})) {
            const previous = totals.get(name)?.metrics;
            totals.set(name, {
              metadata: entry.metadata,
              metrics: previous
                ? {
                    spend: previous.spend + entry.metrics.spend,
                    api_requests: previous.api_requests + entry.metrics.api_requests,
                    total_tokens: previous.total_tokens + entry.metrics.total_tokens,
                    prompt_tokens: sumKnownTokenCounts(previous.prompt_tokens, entry.metrics.prompt_tokens),
                    completion_tokens: sumKnownTokenCounts(previous.completion_tokens, entry.metrics.completion_tokens),
                    cache_read_input_tokens: sumKnownTokenCounts(
                      previous.cache_read_input_tokens,
                      entry.metrics.cache_read_input_tokens,
                    ),
                    cache_creation_input_tokens: sumKnownTokenCounts(
                      previous.cache_creation_input_tokens,
                      entry.metrics.cache_creation_input_tokens,
                    ),
                  }
                : entry.metrics,
            });
          }
        }
        return [...totals.entries()]
          .sort(([, a], [, b]) => b.metrics.spend - a.metrics.spend)
          .slice(0, 20)
          .map(([name, entry]) => ({
            name: dimension === "api_keys" ? hash.parse(name) ?? "[redacted]" : text.parse(name),
            alias: entry.metadata?.key_alias ?? entry.metadata?.team_alias,
            ...entry.metrics,
          }));
      };
      return {
        totals: metadata,
        daily: results.map(({ date, metrics }) => ({ date, ...metrics })),
        breakdown_limit: 20,
        top_models: top("models"),
        top_providers: top("providers"),
        top_entities: top("entities"),
        top_api_keys: top("api_keys"),
      };
    });
  const teamEnvelope = {
    ...pagination,
    teams: limitedList(team).optional(),
    team_info: team.optional(),
    keys: limitedList(key).optional(),
    members: limitedList(member).optional(),
  };
  const schemas = {
    key: key.extend({ ...pagination, keys: limitedList(z.union([key, hash])).optional(), info: key.optional() }),
    team: team.extend(teamEnvelope),
    user: user.extend({ ...pagination, users: limitedList(user).optional(), user_info: user.optional() }),
    budget: z.union([
      limitedList(budget),
      budget.extend({ data: limitedList(budget).optional(), meta: z.object(pagination).optional() }),
    ]),
    spend: limitedList(spend),
    usage,
    log: z.object({ ...pagination, data: limitedList(log) }),
  };
  const parsed = schemas[kind].safeParse(value);
  if (!parsed.success)
    return { notice: "The gateway returned an unsupported result shape. Open the resource to inspect it." };
  if (JSON.stringify(parsed.data).length > 24_000) {
    return { notice: "The result is too large to summarize safely. Use a smaller page or narrower filters." };
  }
  return parsed.data;
}
