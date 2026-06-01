/**
 * Multi-turn example — send a long-running prompt and queue a follow-up
 * message into the same run.
 *
 * Run with:
 *   LITELLM_API_KEY=sk-... LITELLM_BASE_URL=http://localhost:4000 \
 *     npx tsx examples/followup.ts
 */

import { Agent } from "../src/index.js";

async function main(): Promise<void> {
  const agent = await Agent.create({
    apiKey: process.env.LITELLM_API_KEY,
    baseUrl: process.env.LITELLM_BASE_URL,
    name: "followup-example",
    model: { id: "claude-4.6-sonnet" },
  });

  // `await using` automatically tears down the VM at scope exit.
  await using session = await agent.createSession({
    repos: [
      { url: "https://github.com/example/repo", startingRef: "main" },
    ],
  });

  const run = await session.send(
    "Implement the new search endpoint described in issue #42."
  );

  // While the run is in flight, queue a clarification.
  setTimeout(() => {
    session
      .followup("Also handle the empty-input case and add a test for it.")
      .catch((e) => console.error("followup failed:", e));
  }, 2_000);

  for await (const event of run.stream()) {
    if (event.type === "delta") {
      const data = event.data as { text?: string };
      if (data.text) process.stdout.write(data.text);
    } else {
      console.log(`\n[${event.seq}] ${event.type}`);
    }
  }

  const result = await run.wait();
  console.log("\n--- run finished:", result.status, "---");
  if (result.git?.branches?.length) {
    for (const b of result.git.branches) {
      console.log("branch:", b.branch, b.prUrl ?? "(no PR)");
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
