# @litellm/agent-sdk

TypeScript SDK for the LiteLLM Agents API. Mirrors the Cursor SDK's style but
fixes its conflation of agent definition and VM session by exposing a clean
3-level hierarchy:

```
Agent       — definition (model, system prompt, tools)
  └─ Session   — VM sandbox running under an agent
       └─ Run      — single execution within a session
```

## Install

```bash
npm i @litellm/agent-sdk
# or
pnpm add @litellm/agent-sdk
```

Requires Node 18+ (Node 20+ recommended for `await using` support).

## Quickstart

```ts
import { Agent } from "@litellm/agent-sdk";

// 1. Define an agent (do this once at app startup).
const agent = await Agent.create({
  apiKey: process.env.LITELLM_API_KEY,
  baseUrl: process.env.LITELLM_BASE_URL, // e.g. http://localhost:4000
  name: "shin-cursor",
  model: { id: "claude-4.6-sonnet" },
  systemPrompt: "You are a senior engineer fixing GitHub issues.",
});

// 2. Spin up a session per workflow (per Slack issue, per ticket, etc).
const session = await agent.createSession({
  repos: [{ url: "https://github.com/example/repo", startingRef: "main" }],
  envVars: { GITHUB_TOKEN: process.env.GITHUB_TOKEN! },
});

// 3. Kick off a run and stream the events.
const run = await session.send("Fix the bug described in issue #123.");
for await (const event of run.stream()) {
  console.log(event.type, event.data);
}

// 4. Wait for the final result.
const result = await run.wait();
console.log(result.status, result.result);

// 5. Tear down the VM when you're done.
await session.delete();
```

## Multi-turn (followups)

```ts
const run = await session.send("Implement the feature.");
const stream = run.stream();

// In another async context, queue a follow-up message into the active run.
await session.followup("Also handle the empty-input edge case, please.");

for await (const event of stream) {
  // events from the original run continue to flow, including any
  // tool calls / deltas resulting from the follow-up.
}
```

## Resumable streams

Every event carries a monotonic `seq`. Pass `startingSeq` to reconnect from a
specific point — useful if your process restarts mid-stream:

```ts
let lastSeq = 0;
for await (const event of run.stream({ startingSeq: lastSeq + 1 })) {
  lastSeq = event.seq;
  // ...
}
```

The SDK also auto-reconnects internally when the underlying socket drops. It
sends both `Last-Event-ID: <seq>` (per the SSE spec) and
`?starting_seq=<seq+1>` (LiteLLM-specific); the proxy honors whichever it
understands and replays from there. Already-seen events are de-duped.

## `await using` (Symbol.asyncDispose)

If you're on TypeScript 5.2+ / Node 20+, you can scope a session to a block:

```ts
{
  await using session = await agent.createSession();
  const run = await session.send("...");
  await run.wait();
} // session is DELETEd automatically here
```

`session.terminate()` is a manual alias for `delete()` for older runtimes.

## Reusing a single agent across sessions

```ts
const agent = await Agent.create({ /* ... */ });

// Same agent definition, three independent VM sessions.
const [s1, s2, s3] = await Promise.all([
  agent.createSession(),
  agent.createSession(),
  agent.createSession(),
]);
```

## Authentication

```ts
const agent = await Agent.create({
  apiKey: "sk-...", // or set LITELLM_API_KEY in the environment
  baseUrl: "https://api.litellm.ai", // default; override for your deployment
  // ...
});
```

`apiKey` falls back to `process.env.LITELLM_API_KEY` if not provided. The SDK
sends it as `Authorization: Bearer <apiKey>`.

## Errors

All errors are instances of `LiteLLMAgentError`:

```ts
import { LiteLLMAgentError } from "@litellm/agent-sdk";

try {
  await agent.createSession();
} catch (e) {
  if (e instanceof LiteLLMAgentError) {
    console.error(e.code, e.status, e.retryable);
  }
}
```

| Code              | Meaning                                       |
| ----------------- | --------------------------------------------- |
| `missing_api_key` | No `apiKey` provided and `LITELLM_API_KEY` unset |
| `not_found`       | Agent / session / run does not exist          |
| `session_busy`    | `send()` called while a run is in flight (409) |
| `rate_limited`    | 429 — SDK retried `maxRetries` times          |
| `timeout`         | Request exceeded `timeoutMs`                  |
| `http_<status>`   | Generic HTTP error fallback                   |

5xx and 429 are retried automatically with exponential backoff. Configure
`maxRetries` and `timeoutMs` per call via `ClientOptions`.

## Configuration

| Option       | Default                       | Purpose                              |
| ------------ | ----------------------------- | ------------------------------------ |
| `apiKey`     | `process.env.LITELLM_API_KEY` | Bearer token for the proxy           |
| `baseUrl`    | `https://api.litellm.ai`      | Proxy URL                            |
| `fetch`      | `globalThis.fetch`            | Override for tests / custom transport |
| `timeoutMs`  | `60_000`                      | Per-request timeout                  |
| `maxRetries` | `3`                           | Retries on 5xx/429/network errors    |

## API reference

### `Agent`

- `Agent.create(options: AgentCreateOptions): Promise<AgentHandle>`
- `Agent.get(agentId: string, options: ClientOptions): Promise<AgentHandle>`
- `Agent.list(options?: ClientOptions & ListOptions): Promise<ListResult<AgentInfo>>`

### `AgentHandle`

- `agent.id: string`
- `agent.name: string`
- `agent.createSession(options?: CreateSessionOptions): Promise<SessionHandle>`
- `agent.getSession(sessionId: string): Promise<SessionHandle>`
- `agent.listSessions(options?: ListOptions): Promise<ListResult<SessionInfo>>`
- `agent.update(patch: Partial<AgentCreateOptions>): Promise<void>`
- `agent.delete(): Promise<void>` — cascades to sessions

### `SessionHandle`

- `session.id: string`
- `session.agentId: string`
- `session.status: SessionStatus`
- `session.send(input: string | { text; images? }): Promise<Run>` — 409 if a run is in flight
- `session.followup(message: string): Promise<void>` — queues into the active run
- `session.getRun(runId: string): Promise<Run>`
- `session.listRuns(options?: ListOptions): Promise<ListResult<Run>>`
- `session.conversation(): Promise<ConversationTurn[]>`
- `session.delete(): Promise<void>` / `session.terminate()`
- `session[Symbol.asyncDispose]()` — for `await using`

### `Run`

- `run.id: string`
- `run.sessionId: string`
- `run.status: RunStatus`
- `run.result: string | null`
- `run.git?: { branches: { branch; prUrl }[] }`
- `run.stream(opts?: { startingSeq?, signal? }): AsyncIterable<RunEvent>`
- `run.wait(): Promise<RunResult>`
- `run.cancel(): Promise<void>`
- `run.conversation(): Promise<ConversationTurn[]>`

## Examples

See [`examples/basic.ts`](./examples/basic.ts) and
[`examples/followup.ts`](./examples/followup.ts).

## License

MIT — © BerriAI
