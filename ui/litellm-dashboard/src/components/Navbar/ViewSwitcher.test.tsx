import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import ViewSwitcher from "./ViewSwitcher";

const { mockUsePluginMode, state } = vi.hoisted(() => {
  const state = {
    mode: "ai-gateway" as string,
    setMode: vi.fn(),
    plugins: [] as { name: string; display_name: string; url: string }[],
    activePlugin: null as { name: string; display_name: string; url: string } | null,
  };
  return { mockUsePluginMode: vi.fn(() => state), state };
});

vi.mock("@/contexts/PluginModeContext", () => ({ usePluginMode: mockUsePluginMode }));

describe("ViewSwitcher", () => {
  afterEach(() => {
    state.mode = "ai-gateway";
    state.plugins = [];
    state.setMode.mockClear();
  });

  it("renders nothing when there are no plugins", () => {
    const { container } = render(<ViewSwitcher />);
    expect(container.firstChild).toBeNull();
  });

  it("labels the button from the active plugin's display_name and lists AI Gateway + each plugin", async () => {
    state.plugins = [
      { name: "litellm-platform-plugin", display_name: "Chat UI", url: "http://localhost:3300" },
      { name: "obs", display_name: "Observability", url: "http://localhost:9000" },
    ];
    state.mode = "litellm-platform-plugin";
    render(<ViewSwitcher />);

    expect(screen.getByRole("button")).toHaveTextContent("Chat UI");
    expect(screen.queryByText("Agent Control Plane")).not.toBeInTheDocument();

    act(() => {
      fireEvent.click(screen.getByRole("button"));
    });
    await waitFor(() => expect(screen.getByText("AI Gateway")).toBeInTheDocument());
    expect(screen.getByText("Observability")).toBeInTheDocument();
  });

  it("switches to AI Gateway when that entry is picked", async () => {
    state.plugins = [{ name: "litellm-platform-plugin", display_name: "Chat UI", url: "http://localhost:3300" }];
    state.mode = "litellm-platform-plugin";
    render(<ViewSwitcher />);

    act(() => {
      fireEvent.click(screen.getByRole("button"));
    });
    await waitFor(() => expect(screen.getByText("AI Gateway")).toBeInTheDocument());
    act(() => {
      fireEvent.click(screen.getByText("AI Gateway"));
    });
    expect(state.setMode).toHaveBeenCalledWith("ai-gateway");
  });

  it("switches to a plugin by name when its entry is picked", async () => {
    state.plugins = [{ name: "litellm-platform-plugin", display_name: "Chat UI", url: "http://localhost:3300" }];
    state.mode = "ai-gateway";
    render(<ViewSwitcher />);

    act(() => {
      fireEvent.click(screen.getByRole("button"));
    });
    await waitFor(() => expect(screen.getByText("Chat UI")).toBeInTheDocument());
    act(() => {
      fireEvent.click(screen.getByText("Chat UI"));
    });
    expect(state.setMode).toHaveBeenCalledWith("litellm-platform-plugin");
  });
});
