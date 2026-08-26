import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import PromptInfoView from "./prompt_info";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getPromptInfo: vi.fn(),
  getPromptVersions: vi.fn(),
  deletePromptCall: vi.fn(),
}));

vi.mock("./prompt_editor_view/PromptCodeSnippets", () => ({
  default: () => <div data-testid="prompt-code-snippets" />,
}));

const promptWithoutTemplate = {
  prompt_spec: {
    prompt_id: "support-reply",
    version: 1,
    litellm_params: { prompt_id: "support-reply" },
    prompt_info: { prompt_type: "dotprompt" },
    created_by: "admin",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  },
  raw_prompt_template: null,
  environments: [],
};

describe("PromptInfoView tabs", () => {
  beforeEach(() => {
    vi.mocked(networking.getPromptInfo).mockReset().mockResolvedValue(promptWithoutTemplate);
    vi.mocked(networking.getPromptVersions).mockReset().mockResolvedValue({ prompts: [] });
  });

  it("shows the raw API response for a prompt that has no template", async () => {
    const user = userEvent.setup();
    render(<PromptInfoView promptId="support-reply" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    expect(await screen.findByRole("tab", { name: "Raw JSON" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Prompt Template" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Raw JSON" }));

    expect(screen.getByText("Raw API Response")).toBeVisible();
    expect(screen.getByText(/"prompt_id": "support-reply"/)).toBeVisible();
  });
});
