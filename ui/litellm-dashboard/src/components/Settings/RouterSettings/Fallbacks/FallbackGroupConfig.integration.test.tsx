import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { FallbackGroup, FallbackGroupConfig } from "./FallbackGroupConfig";
import { ModelProviders } from "./providerWildcards";

const availableModels = ["claude-sonnet-4-5", "anthropic/claude-opus-4-1", "openai/gpt-4o", "gpt-4o-mini"];
const modelProviders: ModelProviders = { "claude-sonnet-4-5": ["anthropic"], "gpt-4o-mini": ["openai"] };

function Harness({ initial, maxFallbacks = 10 }: { initial: FallbackGroup; maxFallbacks?: number }) {
  const [group, setGroup] = useState(initial);
  return (
    <>
      <FallbackGroupConfig
        group={group}
        onChange={setGroup}
        availableModels={availableModels}
        modelProviders={modelProviders}
        maxFallbacks={maxFallbacks}
      />
      <output data-testid="state">{JSON.stringify(group)}</output>
    </>
  );
}

const state = (): FallbackGroup => JSON.parse(screen.getByTestId("state").textContent ?? "{}");

describe("FallbackGroupConfig provider wildcards", () => {
  it("lets the user pick All OpenAI models as the primary, found by typing openai/*", async () => {
    const user = userEvent.setup();
    render(<Harness initial={{ id: "1", primaryModel: null, fallbackModels: [] }} />);

    await user.type(screen.getByRole("combobox", { name: /primary model/i }), "openai/*");
    await user.click(await screen.findByRole("option", { name: /All OpenAI models/i }));

    expect(state().primaryModel).toBe("openai/*");
  });

  it("expands All Anthropic models into every Anthropic model group in the fallback chain", async () => {
    const user = userEvent.setup();
    render(<Harness initial={{ id: "1", primaryModel: "openai/*", fallbackModels: [] }} />);

    await user.click(screen.getByRole("combobox", { name: "Select fallback models to add..." }));
    await user.click(await screen.findByRole("option", { name: /All Anthropic models/i }));

    expect(state().fallbackModels).toEqual(["claude-sonnet-4-5", "anthropic/claude-opus-4-1"]);
  });

  it("does not offer the primary provider wildcard as its own fallback", async () => {
    const user = userEvent.setup();
    render(<Harness initial={{ id: "1", primaryModel: "openai/*", fallbackModels: [] }} />);

    await user.click(screen.getByRole("combobox", { name: "Select fallback models to add..." }));

    expect(await screen.findByRole("option", { name: /All Anthropic models/i })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /All OpenAI models/i })).not.toBeInTheDocument();
  });
});
