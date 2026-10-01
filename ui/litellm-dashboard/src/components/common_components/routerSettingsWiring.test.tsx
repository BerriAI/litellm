import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import type { FallbackGroup } from "../Settings/RouterSettings/Fallbacks/FallbackGroupConfig";
import type { RouterSettingsFormValue } from "../router_settings/RouterSettingsForm";
import RouterSettingsAccordion from "./RouterSettingsAccordion";
import { routerSettingsEditorValue } from "./routerSettingsPayload";

vi.mock("../networking", () => ({
  getRouterSettingsCall: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-5.5" }, { model_group: "gpt-4o-mini" }]),
  fetchAvailableModelsForTeam: vi.fn().mockResolvedValue([]),
}));

vi.mock("../router_settings/RouterSettingsForm", () => ({
  default: ({ value }: { value: RouterSettingsFormValue }) => (
    <div data-testid="loadbalancing">{JSON.stringify(value.routerSettings)}</div>
  ),
}));

vi.mock("../Settings/RouterSettings/Fallbacks/FallbackSelectionForm", () => ({
  FallbackSelectionForm: ({ groups }: { groups: FallbackGroup[] }) => (
    <div data-testid="fallbacks">
      {groups.map((g) => `${g.primaryModel ?? "none"}->${g.fallbackModels.join("|") || "none"}`).join(" ")}
    </div>
  ),
}));

// Captured verbatim from GET /key/info on a live proxy for a key created through the UI.
// tag_routing_prefix is stored on the key and accepted by /key/update, but the accordion
// has no control for it, so the projection must drop it without losing the rest.
const KEY_INFO_ROUTER_SETTINGS: Record<string, unknown> = {
  fallbacks: [{ "gpt-5.5": ["gpt-4o-mini"] }],
  num_retries: 3,
  tag_routing_prefix: "team-",
};

const renderAccordion = (stored: Record<string, unknown> | null): ReactElement => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <RouterSettingsAccordion accessToken="test-token" value={routerSettingsEditorValue(stored)} />
  </QueryClientProvider>
);

describe("key router settings wiring, /key/info payload through to rendered output", () => {
  it("should prefill both tabs from the stored settings", async () => {
    render(renderAccordion(KEY_INFO_ROUTER_SETTINGS));

    await waitFor(() => {
      expect(screen.getByTestId("fallbacks")).toHaveTextContent("gpt-5.5->gpt-4o-mini");
    });
    expect(JSON.parse(screen.getByTestId("loadbalancing").textContent ?? "{}")).toMatchObject({ num_retries: 3 });
  });

  it("should not leak a field the accordion has no control for into the editor", async () => {
    render(renderAccordion(KEY_INFO_ROUTER_SETTINGS));

    expect(await screen.findByTestId("loadbalancing")).toBeInTheDocument();
    expect(screen.getByTestId("loadbalancing")).not.toHaveTextContent(/tag_routing_prefix/);
  });

  it("should render an empty editor for a key holding only fields it cannot show", async () => {
    render(renderAccordion({ tag_routing_prefix: "team-" }));

    await waitFor(() => expect(screen.getByTestId("fallbacks")).toHaveTextContent("none->none"));
    expect(JSON.parse(screen.getByTestId("loadbalancing").textContent ?? "null")).toEqual({});
  });
});
