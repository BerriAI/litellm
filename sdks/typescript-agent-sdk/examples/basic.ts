/**
 * Basic example — Agent.create -> session.send -> stream -> wait.
 *
 * Run with:
 *   LITELLM_API_KEY=sk-... LITELLM_BASE_URL=http://localhost:4000 \
 *     npx tsx examples/basic.ts
 */

import { Agent } from "../src/index.js";

async function main(): Promise<void> {
  const agent = await Agent.create({
    apiKey: process.env.LITELLM_API_KEY,
    baseUrl: process.env.LITELLM_BASE_URL,
    name: "basic-example",
    model: { id: "claude-4.6-sonnet" },
    systemPrompt: "You are a helpful assistant.",
  });

  console.log("agent created:", agent.id);

  const session = await agent.createSession();
  console.log("session created:", session.id);

  const run = await session.send("Say hi in one short sentence.");
  console.log("run started:", run.id);

  for await (const event of run.stream()) {
    console.log(`[${event.seq}] ${event.type}`, event.data);
  }

  const result = await run.wait();
  console.log("final status:", result.status);
  console.log("result:", result.result);

  await session.delete();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
