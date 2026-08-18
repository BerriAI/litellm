import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import EditFallbacks, { Fallbacks } from "./EditFallbacks";
import * as fetchModelsModule from "@/components/llm_calls/fetch_models";

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

const renderWithQueryClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

describe("EditFallbacks", () => {
  const accessToken = "test-token";
  const fallbackEntry = { "gpt-4": ["gpt-3.5-turbo", "claude-3-opus"] };
  const value: Fallbacks = [{ "gpt-4": ["gpt-3.5-turbo", "claude-3-opus"] }, { "claude-3-opus": ["gpt-4"] }];

  const setup = (overrides: Partial<React.ComponentProps<typeof EditFallbacks>> = {}) => {
    const onChange = overrides.onChange ?? vi.fn().mockResolvedValue(undefined);
    const onClose = overrides.onClose ?? vi.fn();
    renderWithQueryClient(
      <EditFallbacks
        accessToken={accessToken}
        fallbackEntry={fallbackEntry}
        value={value}
        onChange={onChange}
        onClose={onClose}
        {...overrides}
      />,
    );
    return { onChange, onClose };
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchModelsModule.fetchAvailableModels).mockResolvedValue([
      { model_group: "gpt-4", mode: "chat" },
      { model_group: "gpt-3.5-turbo", mode: "chat" },
      { model_group: "claude-3-opus", mode: "chat" },
      { model_group: "gemini-pro", mode: "chat" },
    ]);
  });

  it("prefills the existing fallback chain for the primary model", async () => {
    setup();
    await waitFor(() => {
      expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
      expect(screen.getByText("claude-3-opus")).toBeInTheDocument();
    });
  });

  it("removes a fallback model and saves only the edited entry", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn().mockResolvedValue(undefined);
    const onClose = vi.fn();
    setup({ onChange, onClose });

    await screen.findByText("gpt-3.5-turbo");
    await user.click(screen.getByTestId("remove-fallback-gpt-3.5-turbo"));

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([{ "gpt-4": ["claude-3-opus"] }, { "claude-3-opus": ["gpt-4"] }]);
    });
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it("blocks saving with an empty fallback chain", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn().mockResolvedValue(undefined);
    setup({ fallbackEntry: { "gpt-4": ["gpt-3.5-turbo"] }, onChange });

    await screen.findByText("gpt-3.5-turbo");
    await user.click(screen.getByTestId("remove-fallback-gpt-3.5-turbo"));

    const saveButton = screen.getByRole("button", { name: /save changes/i });
    expect(saveButton).toBeDisabled();
    expect(onChange).not.toHaveBeenCalled();
  });
});
