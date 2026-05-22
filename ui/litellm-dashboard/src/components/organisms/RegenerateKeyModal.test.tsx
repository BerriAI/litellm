import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { RegenerateKeyModal } from "./RegenerateKeyModal";
import { KeyResponse } from "../key_team_helpers/key_list";

// Mock the networking call
const mockRegenerateKeyCall = vi.fn();
vi.mock("../networking", () => ({
  regenerateKeyCall: (...args: unknown[]) => mockRegenerateKeyCall(...args),
}));

const makeToken = (overrides: Partial<KeyResponse> = {}): KeyResponse =>
  ({
    token: "token-hash-123",
    token_id: "token-id-123",
    key_name: "sk-test-key",
    key_alias: "my-test-key",
    max_budget: 100,
    tpm_limit: 5000,
    rpm_limit: 500,
    duration: "30d",
    expires: "2026-12-31T00:00:00Z",
    ...overrides,
  }) as KeyResponse;

describe("RegenerateKeyModal", () => {
  const mockOnClose = vi.fn();
  const mockOnKeyUpdate = vi.fn();

  const defaultProps = {
    selectedToken: makeToken(),
    visible: true,
    onClose: mockOnClose,
    onKeyUpdate: mockOnKeyUpdate,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the modal with correct title", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    expect(screen.getByText("Regenerate Virtual Key")).toBeInTheDocument();
  });

  it("should not render the modal when visible is false", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} visible={false} />);
    expect(screen.queryByText("Regenerate Virtual Key")).not.toBeInTheDocument();
  });

  it("should display the form with pre-filled values", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);

    const keyAliasInput = screen.getByLabelText("Key Alias") as HTMLInputElement;
    expect(keyAliasInput).toBeDisabled();
    expect(keyAliasInput).toHaveValue("my-test-key");
  });

  it("should display the current expiry when token has expires", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    expect(screen.getByText(/Current expiry:/)).toBeInTheDocument();
  });

  it("should display 'Never' when token has no expires", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} selectedToken={makeToken({ expires: undefined })} />);
    expect(screen.getByText("Current expiry: Never")).toBeInTheDocument();
  });

  it("should show Cancel and Regenerate buttons in form view", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Regenerate/ })).toBeInTheDocument();
  });

  it("should call onClose when Cancel is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(mockOnClose).toHaveBeenCalledOnce();
  });

  it("should call onClose when the X close button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(mockOnClose).toHaveBeenCalledOnce();
  });

  it("should render form fields for budget and rate limits", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);

    expect(screen.getByText("Max Budget (USD)")).toBeInTheDocument();
    expect(screen.getByText("TPM Limit")).toBeInTheDocument();
    expect(screen.getByText("RPM Limit")).toBeInTheDocument();
  });

  it("should render duration and grace period fields", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);

    expect(screen.getByText("Expire Key")).toBeInTheDocument();
    expect(screen.getByText("Grace Period")).toBeInTheDocument();
  });

  it("should display grace period recommendation text", () => {
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    expect(screen.getByText("Recommended: 24h to 72h for production keys")).toBeInTheDocument();
  });

  it("should call regenerateKeyCall and show success view on successful regeneration", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(mockRegenerateKeyCall).toHaveBeenCalledOnce();
    });

    await waitFor(() => {
      expect(screen.getByText("sk-new-regenerated-key")).toBeInTheDocument();
    });

    expect(screen.getByText(/will not see it again/)).toBeInTheDocument();
  });

  it("should show Close button after successful regeneration", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(screen.getByText("sk-new-regenerated-key")).toBeInTheDocument();
    });

    // Should show Close buttons (footer + modal X), not Cancel/Regenerate
    const closeButtons = screen.getAllByRole("button", { name: "Close" });
    expect(closeButtons.length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Regenerate/ })).not.toBeInTheDocument();
  });

  it("should show Copy Key button after successful regeneration", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Copy Key/ })).toBeInTheDocument();
    });
  });

  it("should swap the Copy Key button to 'Copied' after clicking it", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    const copyButton = await screen.findByRole("button", { name: /Copy Key/ });
    await user.click(copyButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Copied/ })).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /Copy Key/ })).not.toBeInTheDocument();
  });

  it("should display the 'Virtual Key' label above the key in the success view", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(screen.getByText("Virtual Key")).toBeInTheDocument();
    });
  });

  it("should call onKeyUpdate with updated data after successful regeneration", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(mockOnKeyUpdate).toHaveBeenCalledOnce();
    });

    const updateCall = mockOnKeyUpdate.mock.calls[0][0];
    expect(updateCall.key_name).toBe("sk-new-regenerated-key");
  });

  it.each([
    ["30s", /New expiry:/],
    ["15m", /New expiry:/],
    ["2h", /New expiry:/],
    ["7d", /New expiry:/],
    ["2w", /New expiry:/],
    ["1mo", /New expiry:/],
  ])("should compute a new expiry preview for duration '%s'", async (durationInput, expected) => {
    const user = userEvent.setup();
    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);

    const durationField = screen.getByPlaceholderText("e.g. 30s, 30h, 30d");
    await user.clear(durationField);
    await user.type(durationField, durationInput);

    await waitFor(() => {
      expect(screen.getByText(expected)).toBeInTheDocument();
    });
  });

  it("should fall back to the previous expiry when duration is unparseable", async () => {
    // Regression: if calculateNewExpiryTime returns null (unrecognised suffix),
    // the payload should fall back to the previous expires rather than null.
    const user = userEvent.setup();
    const previousExpires = "2026-12-31T00:00:00Z";
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(
      <RegenerateKeyModal {...defaultProps} selectedToken={makeToken({ expires: previousExpires })} />,
    );

    const durationField = screen.getByPlaceholderText("e.g. 30s, 30h, 30d");
    await user.clear(durationField);
    await user.type(durationField, "bogus");

    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(mockOnKeyUpdate).toHaveBeenCalledOnce();
    });

    const updateCall = mockOnKeyUpdate.mock.calls[0][0];
    expect(updateCall.expires).toBe(previousExpires);
  });

  it("should pass form values to onKeyUpdate even when the API echoes back different limits", async () => {
    // Regression: when the regenerate endpoint returns GenerateKeyResponse, it echoes
    // back the existing max_budget / tpm_limit / rpm_limit. The modal must prefer the
    // values the user just submitted, not whatever the server echoes.
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
      // stale values echoed from the server
      max_budget: 9999,
      tpm_limit: 9999,
      rpm_limit: 9999,
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(mockOnKeyUpdate).toHaveBeenCalledOnce();
    });

    const updateCall = mockOnKeyUpdate.mock.calls[0][0];
    // The form's pre-filled values (from makeToken) must win over the API echo.
    expect(updateCall.max_budget).toBe(100);
    expect(updateCall.tpm_limit).toBe(5000);
    expect(updateCall.rpm_limit).toBe(500);
  });

  it("should display key alias in success view", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(screen.getByText("my-test-key")).toBeInTheDocument();
    });
  });

  it("should display 'No alias set' when key has no alias", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-regenerated-key",
      token: "new-token-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} selectedToken={makeToken({ key_alias: undefined })} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(screen.getByText("No alias set")).toBeInTheDocument();
    });
  });

  it("should not call regenerateKeyCall when selectedToken is null", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RegenerateKeyModal {...defaultProps} selectedToken={null} />);

    // The form shouldn't even be populated, but we check the button doesn't trigger a call
    const regenerateBtn = screen.queryByRole("button", { name: /Regenerate/ });
    if (regenerateBtn) {
      await user.click(regenerateBtn);
    }

    expect(mockRegenerateKeyCall).not.toHaveBeenCalled();
  });

  it("should pass the correct token identifier to regenerateKeyCall", async () => {
    const user = userEvent.setup();
    mockRegenerateKeyCall.mockResolvedValue({
      key: "sk-new-key",
      token: "new-hash",
    });

    renderWithProviders(<RegenerateKeyModal {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /Regenerate/ }));

    await waitFor(() => {
      expect(mockRegenerateKeyCall).toHaveBeenCalledWith(
        "123", // accessToken from mocked useAuthorized
        "token-hash-123", // selectedToken.token
        expect.any(Object),
      );
    });
  });
});
