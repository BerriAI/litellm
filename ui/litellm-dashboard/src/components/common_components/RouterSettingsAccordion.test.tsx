import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RouterSettingsFormValue } from "../router_settings/RouterSettingsForm";
import { modelInfoCall } from "../networking";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./RouterSettingsAccordion";

vi.mock("../networking", () => ({
  getRouterSettingsCall: vi.fn().mockResolvedValue({}),
  modelInfoCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("../Settings/RouterSettings/Fallbacks/FallbackSelectionForm", () => ({
  FallbackSelectionForm: ({ availableModels }: { availableModels: string[] }) => (
    <div data-testid="available-models">{availableModels.join(",")}</div>
  ),
}));

vi.mock("@tremor/react", () => ({
  TabGroup: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  TabList: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Tab: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  TabPanels: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  TabPanel: ({ children }: { children: ReactNode }) => <div>{children}</div>,
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

describe("RouterSettingsAccordion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanup();
    if (vi.isFakeTimers()) {
      vi.runOnlyPendingTimers();
    }
    vi.useRealTimers();
  });

  const flushPromises = async () => {
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
  };

  const flushInitialPropagation = async (onChange: ReturnType<typeof vi.fn>) => {
    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    onChange.mockClear();
  };

  it("debounces propagation and calls onChange once with the last value", async () => {
    const onChange = vi.fn<(value: RouterSettingsAccordionValue) => void>();
    render(<RouterSettingsAccordion accessToken="test-token" onChange={onChange} />);
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

  it("populates the fallback dropdown with team-scoped models when teamId is set", async () => {
    vi.mocked(modelInfoCall).mockResolvedValueOnce({
      data: [{ model_name: "team-a-model" }, { model_name: "team-b-model" }],
    });

    render(<RouterSettingsAccordion accessToken="test-token" teamId="team-1" />);
    await flushPromises();

    expect(screen.getByTestId("available-models").textContent).toBe("team-a-model,team-b-model");
    expect(modelInfoCall).toHaveBeenCalledWith("test-token", "", "", 1, 1000, undefined, undefined, "team-1");
    expect(fetchAvailableModels).not.toHaveBeenCalled();
  });

  it("uses the global model list when no teamId is provided", async () => {
    vi.mocked(fetchAvailableModels).mockResolvedValueOnce([{ model_group: "shared-gpt" }]);

    render(<RouterSettingsAccordion accessToken="test-token" />);
    await flushPromises();

    expect(screen.getByTestId("available-models").textContent).toBe("shared-gpt");
    expect(modelInfoCall).not.toHaveBeenCalled();
  });

  it("does not call onChange when unmounted mid-wait", async () => {
    const onChange = vi.fn<(value: RouterSettingsAccordionValue) => void>();
    const { unmount } = render(<RouterSettingsAccordion accessToken="test-token" onChange={onChange} />);
    await flushInitialPropagation(onChange);

    fireEvent.click(screen.getByText("set-least-busy"));
    unmount();

    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(onChange).not.toHaveBeenCalled();
  });
});
