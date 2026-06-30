import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReactNode } from "react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import UpdateModelCredentialsModal from "./update_model_credentials_modal";
import { Providers } from "./provider_info_helpers";
import * as networking from "./networking";

vi.mock("./networking", async () => {
  const actual = await vi.importActual("./networking");
  return {
    ...actual,
    getProviderCreateMetadata: vi.fn().mockResolvedValue([
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        credential_fields: [
          { key: "api_base", label: "API Base", field_type: "text", placeholder: "https://api.openai.com/v1" },
          { key: "api_key", label: "OpenAI API Key", field_type: "password", required: true },
        ],
      },
    ]),
    modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
  };
});

vi.mock("./molecules/notifications_manager", () => ({
  default: { success: vi.fn(), error: vi.fn(), info: vi.fn(), fromBackend: vi.fn() },
}));

const mockModelPatchUpdateCall = vi.mocked(networking.modelPatchUpdateCall);

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

const createQueryClient = () => new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });

const renderModal = (overrides: Partial<Parameters<typeof UpdateModelCredentialsModal>[0]> = {}) => {
  const queryClient = createQueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return render(
    <UpdateModelCredentialsModal
      open
      onCancel={vi.fn()}
      accessToken="test-token"
      modelId="model-123"
      provider={Providers.OpenAI}
      onUpdated={vi.fn()}
      {...overrides}
    />,
    { wrapper },
  );
};

describe("UpdateModelCredentialsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends a minimal PATCH with only the fields the user typed", async () => {
    const user = userEvent.setup();
    const onUpdated = vi.fn();
    const onCancel = vi.fn();
    renderModal({ onUpdated, onCancel });

    const apiKey = await screen.findByLabelText("OpenAI API Key");
    await user.type(apiKey, "sk-rotated-9988");
    await user.click(screen.getByRole("button", { name: /update credentials/i }));

    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    const [token, payload, modelId] = mockModelPatchUpdateCall.mock.calls[0];
    expect(token).toBe("test-token");
    expect(modelId).toBe("model-123");
    // Exact match: only the typed field is sent; the blank api_base is never included.
    expect(payload).toEqual({ litellm_params: { api_key: "sk-rotated-9988" }, model_info: { id: "model-123" } });
    expect(onUpdated).toHaveBeenCalledTimes(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("does not call the update API when every field is left blank", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByLabelText("OpenAI API Key");
    await user.click(screen.getByRole("button", { name: /update credentials/i }));

    // Give the submit handler a chance to run before asserting it did nothing.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
  });
});
