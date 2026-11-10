import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./RouterSettingsAccordion";

vi.mock("../networking", () => ({
  getRouterSettingsCall: vi.fn().mockResolvedValue({}),
  modelInfoCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-5" }]),
  fetchAvailableModelsForTeam: vi.fn().mockResolvedValue([{ model_group: "gpt-5" }]),
}));

const EchoingCreateForm = () => {
  const [value, setValue] = useState<RouterSettingsAccordionValue | undefined>(undefined);
  return <RouterSettingsAccordion accessToken="test-token" value={value} onChange={setValue} />;
};

describe("RouterSettingsAccordion on a create form", () => {
  it("should render every reliability input after the parent echoes the first value back", async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <EchoingCreateForm />
      </QueryClientProvider>,
    );
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 250));
    });

    for (const field of [
      "allowed_fails",
      "cooldown_time",
      "num_retries",
      "timeout",
      "retry_after",
      "model_group_alias",
    ]) {
      expect(screen.getByRole("textbox", { name: new RegExp(field) })).toBeEnabled();
    }
  });
});
