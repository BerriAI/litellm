import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import UpdateModelCredentialsModal from "./update_model_credentials_modal";
import * as networking from "./networking";

vi.mock("./networking", async () => {
  const actual = await vi.importActual("./networking");
  return {
    ...actual,
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

const renderModal = (overrides: Partial<Parameters<typeof UpdateModelCredentialsModal>[0]> = {}) =>
  render(
    <UpdateModelCredentialsModal
      open
      onCancel={vi.fn()}
      accessToken="test-token"
      modelId="model-123"
      onUpdated={vi.fn()}
      {...overrides}
    />,
  );

describe("UpdateModelCredentialsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends a minimal PATCH with only the new api_key", async () => {
    const user = userEvent.setup();
    const onUpdated = vi.fn();
    const onCancel = vi.fn();
    renderModal({ onUpdated, onCancel });

    await user.type(screen.getByLabelText(/new api key/i), "sk-rotated-9988");
    await user.click(screen.getByRole("button", { name: /update api key/i }));

    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    const [token, payload, modelId] = mockModelPatchUpdateCall.mock.calls[0];
    expect(token).toBe("test-token");
    expect(modelId).toBe("model-123");
    // Exactly the new key plus the id — nothing else from the deployment.
    expect(payload).toEqual({ litellm_params: { api_key: "sk-rotated-9988" }, model_info: { id: "model-123" } });
    expect(onUpdated).toHaveBeenCalledTimes(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("does not call the update API when the field is left blank", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: /update api key/i }));

    // Required-field validation blocks submit; give it a tick then assert no call.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
  });
});
