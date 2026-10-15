import { describe, expect, it } from "vitest";
import { issueLabel } from "./issueLabel";

describe("issueLabel", () => {
  it("names the Input field for both endpoints instead of the JSON key", () => {
    expect(issueLabel("state")).toBe("Input");
    expect(issueLabel("input")).toBe("Input");
  });

  it("names top-level fields and the request itself", () => {
    expect(issueLabel("model")).toBe("Model");
    expect(issueLabel("questions")).toBe("Questions");
    expect(issueLabel("safety_identifier")).toBe("Safety identifier");
    expect(issueLabel("root")).toBe("Request");
    expect(issueLabel("syntax")).toBe("Request");
  });

  it("names a question by its key or its 1-based position and the field inside it", () => {
    expect(issueLabel("questions.urgency")).toBe('Question "urgency"');
    expect(issueLabel("questions.urgency.criteria")).toBe('Question "urgency" options');
    expect(issueLabel("questions.urgency.criteria.1")).toBe('Question "urgency" options');
    expect(issueLabel("questions.category.instructions")).toBe('Question "category" instructions');
    expect(issueLabel("questions.0.name")).toBe("Question 1 name");
    expect(issueLabel("questions.2.type")).toBe("Question 3 answer type");
  });

  it("falls back to the humanized key for anything unknown", () => {
    expect(issueLabel("max_tokens")).toBe("max tokens");
    expect(issueLabel("questions.a.max_items")).toBe('Question "a" max items');
  });
});
