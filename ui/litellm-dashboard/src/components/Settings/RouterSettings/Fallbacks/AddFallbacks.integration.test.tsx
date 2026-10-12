import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddFallbacks from "./AddFallbacks";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";

vi.mock("@/components/llm_calls/fetch_models", () => ({ fetchAvailableModels: vi.fn() }));

const models = [
  { model_group: "primary-fast", providers: ["openai"] },
  { model_group: "primary-writer", providers: ["openai"] },
  { model_group: "backup-fast", providers: ["anthropic"] },
  { model_group: "backup-writer", providers: ["anthropic"] },
];

describe("provider-wide fallback form", () => {
  beforeEach(() => {
    vi.mocked(fetchAvailableModels).mockResolvedValue(models);
  });

  it("previews aliases, displays logos and saves reordered concrete chains for every primary", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn().mockResolvedValue(undefined);
    render(<AddFallbacks accessToken="test-token" value={[{ untouched: ["backup-fast"] }]} onChange={onChange} />);
    await user.click(screen.getByRole("button", { name: /Add Fallbacks/ }));
    await user.click(screen.getByRole("combobox", { name: /Primary Model/ }));
    await user.click(await screen.findByRole("option", { name: /All OpenAI Models/ }));

    const primaries = screen.getByRole("list", { name: "Primary models" });
    expect(within(primaries).getByText("primary-fast")).toBeInTheDocument();
    expect(within(primaries).getByText("primary-writer")).toBeInTheDocument();
    expect(within(primaries).getAllByRole("img", { name: "openai logo" })).toHaveLength(2);
    expect(screen.getByText(/Newly added models and wildcard routes are not included/)).toBeInTheDocument();

    await user.click(screen.getByRole("combobox", { name: /Select fallback models to add/ }));
    await user.click(await screen.findByRole("option", { name: /All Anthropic Models/ }));
    await user.keyboard("{Escape}");
    const chain = screen.getByRole("list", { name: "Fallback chain" });
    expect(within(chain).getAllByRole("img", { name: "anthropic logo" })).toHaveLength(2);
    await user.click(within(chain).getByRole("button", { name: "Move backup-writer earlier" }));
    await user.click(screen.getByRole("button", { name: "Save All Configurations" }));
    expect(onChange).toHaveBeenCalledWith([
      { untouched: ["backup-fast"] },
      { "primary-fast": ["backup-writer", "backup-fast"] },
      { "primary-writer": ["backup-writer", "backup-fast"] },
    ]);
  });
});
