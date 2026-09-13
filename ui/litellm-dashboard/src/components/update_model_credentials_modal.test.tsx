import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import UpdateModelCredentialsModal from "./update_model_credentials_modal";
import * as networking from "./networking";
import { toast } from "@/lib/toast";

vi.mock("./networking", async () => {
  const actual = await vi.importActual("./networking");
  return {
    ...actual,
    modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
  };
});

const mockModelPatchUpdateCall = vi.mocked(networking.modelPatchUpdateCall);
const mockToast = vi.mocked(toast);

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

    fireEvent.change(screen.getByLabelText(/new api key/i), { target: { value: "sk-rotated-9988" } });
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

  it("renders the required message when the field is left blank", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: /update api key/i }));

    expect(await screen.findByText("Enter a new API key")).toBeInTheDocument();
  });

  it("rejects a whitespace-only key without sending a PATCH", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText(/new api key/i), { target: { value: "   " } });
    await user.click(screen.getByRole("button", { name: /update api key/i }));

    await waitFor(() => expect(mockToast.fromError).toHaveBeenCalledWith("Enter a new API key"));
    expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
  });

  it("trims surrounding whitespace off the key it sends", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText(/new api key/i), { target: { value: "  sk-pad-77  " } });
    await user.click(screen.getByRole("button", { name: /update api key/i }));

    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(mockModelPatchUpdateCall.mock.calls[0][1]).toEqual({
      litellm_params: { api_key: "sk-pad-77" },
      model_info: { id: "model-123" },
    });
  });

  it("submits on Enter from inside the key field", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByLabelText(/new api key/i), "sk-enter-1{Enter}");

    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(mockModelPatchUpdateCall.mock.calls[0][1]).toEqual({
      litellm_params: { api_key: "sk-enter-1" },
      model_info: { id: "model-123" },
    });
  });

  it("reveals and re-hides the key without touching the value", async () => {
    const user = userEvent.setup();
    renderModal();

    const field = screen.getByLabelText(/new api key/i);
    fireEvent.change(field, { target: { value: "sk-peek-42" } });
    expect(field).toHaveAttribute("type", "password");

    await user.click(screen.getByRole("button", { name: /show password/i }));
    expect(field).toHaveAttribute("type", "text");
    expect(field).toHaveValue("sk-peek-42");

    await user.click(screen.getByRole("button", { name: /hide password/i }));
    expect(field).toHaveAttribute("type", "password");
    expect(field).toHaveValue("sk-peek-42");
  });
});
