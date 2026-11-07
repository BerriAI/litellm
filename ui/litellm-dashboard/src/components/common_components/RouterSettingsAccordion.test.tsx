import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createRef, useState, type ReactElement, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RouterSettingsFormValue } from "../router_settings/RouterSettingsForm";
import { fetchAvailableModels, fetchAvailableModelsForTeam } from "@/components/llm_calls/fetch_models";
import { getRouterSettingsCall, modelInfoCall } from "../networking";
import RouterSettingsAccordion, {
  RouterSettingsAccordionRef,
  RouterSettingsAccordionValue,
} from "./RouterSettingsAccordion";

vi.mock("../networking", () => ({
  getRouterSettingsCall: vi.fn().mockResolvedValue({}),
  modelInfoCall: vi.fn(),
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

  describe("deployment weights", () => {
    const gpt5Deployments = {
      data: [
        { model_name: "gpt-5", litellm_params: { model: "openai/gpt-5" }, model_info: { id: "dep-openai" } },
        { model_name: "gpt-5", litellm_params: { model: "azure/gpt-5" }, model_info: { id: "dep-azure" } },
      ],
    };

    beforeEach(() => {
      vi.mocked(modelInfoCall).mockResolvedValue(gpt5Deployments);
    });

    it("should save stored weights together with a newly entered one", async () => {
      const ref = createRef<RouterSettingsAccordionRef>();
      renderWithQueryClient(
        <RouterSettingsAccordion
          ref={ref}
          accessToken="test-token"
          teamId="team-123"
          value={{ router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } } }}
        />,
      );

      fireEvent.change(await screen.findByLabelText("Weight for azure/gpt-5 (dep-azure)"), { target: { value: "1" } });

      expect(await screen.findByText("90% of traffic")).toBeInTheDocument();
      expect(screen.getByText("10% of traffic")).toBeInTheDocument();
      expect(ref.current?.getValue().router_settings.weights).toStrictEqual({
        "gpt-5": { "dep-openai": 9, "dep-azure": 1 },
      });
      expect(vi.mocked(modelInfoCall).mock.calls[0][7]).toBe("team-123");
      expect(vi.mocked(modelInfoCall).mock.calls[0][11]).toBe("gpt-5");
    });

    it("should save null once every weight is cleared, so the stored weights are removed", async () => {
      const ref = createRef<RouterSettingsAccordionRef>();
      renderWithQueryClient(
        <RouterSettingsAccordion
          ref={ref}
          accessToken="test-token"
          value={{ router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } } }}
        />,
      );

      fireEvent.change(await screen.findByLabelText("Weight for openai/gpt-5 (dep-openai)"), { target: { value: "" } });

      await waitFor(() => expect(ref.current?.getValue().router_settings.weights).toBeNull());
    });

    it("should keep a model group on screen when the parent echoes back the value it was sent", async () => {
      const EchoingParent = () => {
        const [value, setValue] = useState<RouterSettingsAccordionValue>({
          router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } },
        });
        return <RouterSettingsAccordion accessToken="test-token" value={value} onChange={setValue} />;
      };
      renderWithQueryClient(<EchoingParent />);

      fireEvent.change(await screen.findByLabelText("Weight for openai/gpt-5 (dep-openai)"), { target: { value: "" } });
      await act(async () => {
        vi.advanceTimersByTime(200);
      });

      expect(screen.getByLabelText("Weight for openai/gpt-5 (dep-openai)")).toHaveValue("");
      expect(screen.getByRole("button", { name: "Remove weights for gpt-5" })).toBeInTheDocument();
    });

    it("should say the deployments failed to load instead of calling saved weights stale", async () => {
      vi.mocked(modelInfoCall).mockRejectedValue(new Error("403"));
      const ref = createRef<RouterSettingsAccordionRef>();
      renderWithQueryClient(
        <RouterSettingsAccordion
          ref={ref}
          accessToken="test-token"
          value={{ router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } } }}
        />,
      );

      expect(
        await screen.findByText("Could not load this model group's deployments. Saved weights are kept."),
      ).toBeInTheDocument();
      expect(screen.queryByText("Deployment no longer exists")).not.toBeInTheDocument();
      expect(ref.current?.getValue().router_settings.weights).toStrictEqual({ "gpt-5": { "dep-openai": 9 } });
    });

    it("should warn that weights are ignored when the inherited global strategy is not simple-shuffle", async () => {
      vi.mocked(getRouterSettingsCall).mockResolvedValueOnce({
        fields: [{ field_name: "routing_strategy", options: ["simple-shuffle"], field_value: "latency-based-routing" }],
      });
      renderWithQueryClient(
        <RouterSettingsAccordion
          accessToken="test-token"
          value={{ router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } } }}
        />,
      );

      expect(
        await screen.findByText("Weights are ignored with latency-based-routing. Select simple-shuffle to use them."),
      ).toBeInTheDocument();
    });

    it("should accept a decimal weight typed one character at a time", async () => {
      const ref = createRef<RouterSettingsAccordionRef>();
      renderWithQueryClient(
        <RouterSettingsAccordion
          ref={ref}
          accessToken="test-token"
          value={{ router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } } }}
        />,
      );
      const input = await screen.findByLabelText("Weight for azure/gpt-5 (dep-azure)");

      fireEvent.change(input, { target: { value: "1." } });
      expect(input).toHaveValue("1.");
      fireEvent.change(input, { target: { value: "1.5" } });

      expect(input).toHaveValue("1.5");
      expect(ref.current?.getValue().router_settings.weights).toStrictEqual({
        "gpt-5": { "dep-openai": 9, "dep-azure": 1.5 },
      });
    });

    it("should not call saved weights stale while the deployments are still loading", async () => {
      vi.mocked(modelInfoCall).mockReturnValue(new Promise(() => {}));
      renderWithQueryClient(
        <RouterSettingsAccordion
          accessToken="test-token"
          value={{ router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } } }}
        />,
      );

      expect(await screen.findByText("dep-openai")).toBeInTheDocument();
      expect(screen.queryByText("Deployment no longer exists")).not.toBeInTheDocument();
    });

    it("should drop weights picked for one team when the form switches to another team", async () => {
      const ref = createRef<RouterSettingsAccordionRef>();
      const value = { router_settings: { weights: { "gpt-5": { "dep-openai": 9 } } } };
      const { rerender } = renderWithQueryClient(
        <RouterSettingsAccordion ref={ref} accessToken="test-token" teamId="team-a" value={value} />,
      );
      await screen.findByLabelText("Weight for openai/gpt-5 (dep-openai)");

      rerender(<RouterSettingsAccordion ref={ref} accessToken="test-token" teamId="team-b" value={value} />);

      await waitFor(() => expect(ref.current?.getValue().router_settings.weights).toBeNull());
    });

    it("should flag a group whose weights are all zero", async () => {
      renderWithQueryClient(
        <RouterSettingsAccordion
          accessToken="test-token"
          value={{ router_settings: { weights: { "gpt-5": { "dep-openai": 0 } } } }}
        />,
      );

      expect(
        await screen.findByText("Set at least one weight above 0, or remove this model group."),
      ).toBeInTheDocument();
    });
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
