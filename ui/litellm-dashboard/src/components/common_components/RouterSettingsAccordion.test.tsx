import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RouterSettingsFormValue } from "../router_settings/RouterSettingsForm";
import { fetchAvailableModels, fetchAvailableModelsForTeam } from "@/components/llm_calls/fetch_models";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./RouterSettingsAccordion";

vi.mock("../networking", () => ({
  getRouterSettingsCall: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "global-model" }]),
  fetchAvailableModelsForTeam: vi.fn().mockResolvedValue([{ model_group: "openai/*" }, { model_group: "gpt-5" }]),
}));

vi.mock("../Settings/RouterSettings/Fallbacks/FallbackSelectionForm", () => ({
  FallbackSelectionForm: ({ availableModels }: { availableModels: string[] }) => (
    <div data-testid="available-models">{availableModels.join(",")}</div>
  ),
}));

vi.mock("../router_settings/RouterSettingsForm", () => ({
  default: ({
    value,
    onChange,
  }: {
    value: RouterSettingsFormValue;
    onChange: (value: RouterSettingsFormValue) => void;
  }) => (
    <div>
      <button onClick={() => onChange({ ...value, selectedStrategy: "least-busy" })}>set-least-busy</button>
      <button onClick={() => onChange({ ...value, selectedStrategy: "usage-based-routing" })}>set-usage-based</button>
    </div>
  ),
}));

const renderWithQueryClient = (ui: ReactElement) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(ui, {
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
};

describe("RouterSettingsAccordion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  const flushInitialPropagation = async (onChange: ReturnType<typeof vi.fn>) => {
    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    onChange.mockClear();
  };

  it("debounces propagation and calls onChange once with the last value", async () => {
    const onChange = vi.fn<(value: RouterSettingsAccordionValue) => void>();
    renderWithQueryClient(<RouterSettingsAccordion accessToken="test-token" onChange={onChange} />);
    await flushInitialPropagation(onChange);

    fireEvent.click(screen.getByText("set-least-busy"));
    act(() => {
      vi.advanceTimersByTime(50);
    });
    fireEvent.click(screen.getByText("set-usage-based"));

    expect(onChange).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(99);
    });
    expect(onChange).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0].router_settings.routing_strategy).toBe("usage-based-routing");
  });

  it("offers the team's own models, including team-scoped BYOK ones, when a teamId is given", async () => {
    renderWithQueryClient(<RouterSettingsAccordion accessToken="test-token" teamId="team-123" />);

    await waitFor(() => {
      expect(screen.getByTestId("available-models")).toHaveTextContent("gpt-5,openai/*");
    });
    expect(fetchAvailableModelsForTeam).toHaveBeenCalledWith("test-token", "team-123");
    expect(fetchAvailableModels).not.toHaveBeenCalled();
  });

  it("falls back to the proxy-wide model listing when no teamId is given", async () => {
    renderWithQueryClient(<RouterSettingsAccordion accessToken="test-token" />);

    await waitFor(() => {
      expect(screen.getByTestId("available-models")).toHaveTextContent("global-model");
    });
    expect(fetchAvailableModelsForTeam).not.toHaveBeenCalled();
  });

  it("ignores a stale team's model response that resolves after a newer team was selected", async () => {
    const resolvers: ((models: { model_group: string }[]) => void)[] = [];
    vi.mocked(fetchAvailableModelsForTeam).mockImplementation(
      () => new Promise((resolve) => resolvers.push(resolve)) as Promise<{ model_group: string }[]>,
    );

    const { rerender } = renderWithQueryClient(<RouterSettingsAccordion accessToken="test-token" teamId="team-slow" />);
    await waitFor(() => expect(resolvers).toHaveLength(1));

    rerender(<RouterSettingsAccordion accessToken="test-token" teamId="team-fast" />);
    await waitFor(() => expect(resolvers).toHaveLength(2));

    await act(async () => {
      resolvers[1]([{ model_group: "fast-team-model" }]);
      resolvers[0]([{ model_group: "slow-team-model" }]);
    });

    await waitFor(() => {
      expect(screen.getByTestId("available-models")).toHaveTextContent("fast-team-model");
    });
    expect(screen.getByTestId("available-models")).not.toHaveTextContent("slow-team-model");
  });

  it("does not call onChange when unmounted mid-wait", async () => {
    const onChange = vi.fn<(value: RouterSettingsAccordionValue) => void>();
    const { unmount } = renderWithQueryClient(<RouterSettingsAccordion accessToken="test-token" onChange={onChange} />);
    await flushInitialPropagation(onChange);

    fireEvent.click(screen.getByText("set-least-busy"));
    unmount();

    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(onChange).not.toHaveBeenCalled();
  });
});
