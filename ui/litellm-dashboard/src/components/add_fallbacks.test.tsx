import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddFallbacks from "./add_fallbacks";

vi.mock("./networking", () => ({
  setCallbacksCall: vi.fn(),
}));

vi.mock("./playground/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(() =>
    Promise.resolve([
      { model_group: "gpt-4" },
      { model_group: "gpt-3.5-turbo" },
      { model_group: "claude-3-opus" },
      { model_group: "claude-3-sonnet" },
    ]),
  ),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("AddFallbacks", () => {
  const mockAccessToken = "test-token";
  const mockRouterSettings = { fallbacks: [] };
  const mockSetRouterSettings = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the component", () => {
    render(
      <AddFallbacks
        accessToken={mockAccessToken}
        routerSettings={mockRouterSettings}
        setRouterSettings={mockSetRouterSettings}
      />,
    );

    expect(screen.getByRole("button", { name: /Add Fallbacks/i })).toBeInTheDocument();
  });
});
