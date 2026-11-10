import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { fireEvent, renderWithProviders, screen } from "@/../tests/test-utils";

import type { CredentialItem } from "../networking";
import ReuseCredentialsModal from "./reuse_credentials";

const EXISTING_CREDENTIAL: CredentialItem = {
  credential_name: "openai-prod",
  credential_values: { api_key: "sk-stored-value", api_base: "https://api.example.com" },
  credential_info: { custom_llm_provider: "openai" },
};

const renderModal = (existingCredential: CredentialItem | null = EXISTING_CREDENTIAL) => {
  const onAddCredential = vi.fn();
  const onCancel = vi.fn();
  const setIsCredentialModalOpen = vi.fn();
  renderWithProviders(
    <ReuseCredentialsModal
      isVisible
      onCancel={onCancel}
      onAddCredential={onAddCredential}
      existingCredential={existingCredential}
      setIsCredentialModalOpen={setIsCredentialModalOpen}
    />,
  );
  return { onAddCredential, onCancel, setIsCredentialModalOpen };
};

const submit = async (user: ReturnType<typeof userEvent.setup>) =>
  await user.click(screen.getByRole("button", { name: "Reuse Credentials" }));

describe("ReuseCredentialsModal", () => {
  it("submits the typed name alongside every stored credential value", async () => {
    const user = userEvent.setup();
    const { onAddCredential, setIsCredentialModalOpen } = renderModal();

    const nameInput = screen.getByLabelText("Credential Name:");
    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "reused-openai" } });
    await submit(user);

    expect(onAddCredential).toHaveBeenCalledTimes(1);
    expect(onAddCredential).toHaveBeenCalledWith({
      credential_name: "reused-openai",
      api_key: "sk-stored-value",
      api_base: "https://api.example.com",
    });
    expect(setIsCredentialModalOpen).toHaveBeenCalledWith(false);
  });

  it("seeds the name from the existing credential and submits it untouched", async () => {
    const user = userEvent.setup();
    const { onAddCredential } = renderModal();

    expect(screen.getByLabelText("Credential Name:")).toHaveValue("openai-prod");
    await submit(user);

    expect(onAddCredential).toHaveBeenCalledWith({
      credential_name: "openai-prod",
      api_key: "sk-stored-value",
      api_base: "https://api.example.com",
    });
  });

  it("renders the stored values as read-only inputs", () => {
    renderModal();

    expect(screen.getByLabelText("api_key")).toBeDisabled();
    expect(screen.getByLabelText("api_key")).toHaveValue("sk-stored-value");
    expect(screen.getByLabelText("api_base")).toBeDisabled();
  });

  it("blocks the submit and shows the required message when the name is cleared", async () => {
    const user = userEvent.setup();
    const { onAddCredential } = renderModal();

    await user.clear(screen.getByLabelText("Credential Name:"));
    await submit(user);

    expect(await screen.findByText("Credential name is required")).toBeInTheDocument();
    expect(onAddCredential).not.toHaveBeenCalled();
  });

  it("submits only the name when the credential carries no stored values", async () => {
    const user = userEvent.setup();
    const { onAddCredential } = renderModal({
      credential_name: "bare",
      credential_values: {},
      credential_info: {},
    });

    await submit(user);

    expect(onAddCredential).toHaveBeenCalledWith({ credential_name: "bare" });
  });

  it("closes without submitting when Cancel is clicked", async () => {
    const user = userEvent.setup();
    const { onAddCredential, onCancel } = renderModal();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onAddCredential).not.toHaveBeenCalled();
  });

  it("submits on Enter from the name field", async () => {
    const user = userEvent.setup();
    const { onAddCredential } = renderModal();

    await user.type(screen.getByLabelText("Credential Name:"), "{Enter}");

    expect(onAddCredential).toHaveBeenCalledTimes(1);
  });
});
