import { describe, expect, it } from "vitest";
import { investigationHandoffText } from "./agentHandoff";

const base = "https://proxy.test/root/";

describe("investigation agent handoff", () => {
  it("fetches the latest investigation and explains how to inspect findings and original evidence", () => {
    const text = investigationHandoffText(base, "lens-one");
    expect(text).toContain(
      "curl --fail-with-body -sS -H 'Authorization: Bearer ***' 'https://proxy.test/root/lens/lens-one'",
    );
    expect(text).toContain("Inspect jobs[0]");
    expect(text).toContain("/lens/lens-one/runs/{job_id}");
    expect(text).toContain("/lens/lens-one/executions/{execution_id}");
    expect(text).toContain("URL-encoding the complete evidence.execution_id as one path segment");
    expect(text).toContain("Follow next_cursor");
    expect(text).toContain("truncated parts, missing evidence, and access errors");
    expect(text).toContain("Do not paste the key into chat");
    expect(text).toContain("findings are null");
  });

  it("keeps accumulated findings distinct from the latest run", () => {
    const text = investigationHandoffText(base, "lens-one", "all");
    expect(text).toContain("all accumulated findings");
    expect(text).not.toContain("Inspect jobs[0]");
    expect(text).not.toContain("/runs/all");
  });

  it("fetches an explicitly selected historical run as well as current feedback", () => {
    const text = investigationHandoffText(base, "lens/one", "run?two&three");
    expect(text).toContain("'https://proxy.test/root/lens/lens%2Fone/runs/run%3Ftwo%26three'");
    expect(text).toContain("selected run's findings");
    expect(text).toContain("current review status and reason by finding id");
    expect(text).not.toContain("Inspect jobs[0]");
  });

  it("quotes shell data, including a custom auth header, without copying credentials", () => {
    const text = investigationHandoffText(
      "https://proxy.test/a'b/$(printf injected)",
      "lens'one",
      "all",
      "x-litellm-key",
    );
    expect(text).toContain("-H 'x-litellm-key: Bearer ***'");
    expect(text).toContain("'https://proxy.test/a'\"'\"'b/$(printf injected)/lens/lens'\"'\"'one'");
  });
});
