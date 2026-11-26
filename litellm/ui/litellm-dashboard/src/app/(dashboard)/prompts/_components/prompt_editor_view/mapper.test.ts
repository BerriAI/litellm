import { describe, expect, it } from "vitest";
import { toPromptEditorPayload } from "./mapper";
import { promptEditorSchema } from "./schema";

const draft = {
  name: " My Prompt ",
  model: "gpt-4o",
  config: { temperature: 0, max_tokens: 500 },
  tools: [],
  developerMessage: "Use the provided data.",
  messages: [{ role: "user", content: "Summarize {{topic}}" }],
  environment: "staging",
};

describe("prompt editor payload", () => {
  it("publishes the normalized ID, selected environment, and dotprompt text", () => {
    expect(toPromptEditorPayload(promptEditorSchema.parse({ prompt: draft }))).toEqual({
      prompt_id: "_my_prompt_",
      litellm_params: {
        prompt_integration: "dotprompt",
        prompt_id: "_my_prompt_",
        dotprompt_content:
          "---\nmodel: gpt-4o\ntemperature: 0\nmax_tokens: 500\ninput:\n  schema:\n    topic: string\n" +
          "output:\n  format: text\n---\n\nDeveloper: Use the provided data.\n\nUser: Summarize {{topic}}",
      },
      prompt_info: { prompt_type: "db", environment: "staging" },
    });
  });

  it("rejects a whitespace-only publish name", () => {
    expect(promptEditorSchema.safeParse({ prompt: { ...draft, name: "  " } }).success).toBe(false);
  });
});
