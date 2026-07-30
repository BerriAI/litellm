import { act, fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RouterSettingsFormValue } from "../router_settings/RouterSettingsForm";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./RouterSettingsAccordion";

vi.mock("../networking", () => ({
  getRouterSettingsCall: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("../Settings/RouterSettings/Fallbacks/FallbackSelectionForm", () => ({
  FallbackSelectionForm: () => null,
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
    vi.useFakeTimers();
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
