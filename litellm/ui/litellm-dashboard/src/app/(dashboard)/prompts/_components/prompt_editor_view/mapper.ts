import type { z } from "zod";
import { convertToDotPrompt } from "./utils";
import type { promptEditorSchema } from "./schema";

export function toPromptEditorPayload({ prompt }: z.output<typeof promptEditorSchema>) {
  const promptId = prompt.name.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
  return {
    prompt_id: promptId,
    litellm_params: {
      prompt_integration: "dotprompt",
      prompt_id: promptId,
      dotprompt_content: convertToDotPrompt(prompt),
    },
    prompt_info: {
      prompt_type: "db",
      environment: prompt.environment,
    },
  };
}
