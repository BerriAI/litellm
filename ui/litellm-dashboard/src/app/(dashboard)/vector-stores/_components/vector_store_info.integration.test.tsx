import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { credentialListCall, vectorStoreInfoCall, vectorStoreUpdateCall } from "@/components/networking";
import { toast } from "@/lib/toast";

import VectorStoreInfoView from "./vector_store_info";

vi.mock("@/components/networking", () => ({
  vectorStoreInfoCall: vi.fn(),
  vectorStoreUpdateCall: vi.fn(),
  credentialListCall: vi.fn(),
}));

vi.mock("./VectorStoreTester", () => ({ __esModule: true, default: () => null }));

const mockInfo = vi.mocked(vectorStoreInfoCall);
const mockUpdate = vi.mocked(vectorStoreUpdateCall);
const mockCredentials = vi.mocked(credentialListCall);
const mockToast = vi.mocked(toast);

const serverRecord = {
  vector_store_id: "vs-1",
  vector_store_name: "support-docs-store",
  vector_store_description: "Docs for support",
  custom_llm_provider: "bedrock",
  vector_store_metadata: { tier: "gold" },
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-02-02T00:00:00Z",
  litellm_credential_name: "bedrock-prod",
};

const renderView = (editVectorStore: boolean) =>
  render(
    <VectorStoreInfoView
      vectorStoreId="vs-1"
      onClose={vi.fn()}
      accessToken="sk-test"
      is_admin={true}
      editVectorStore={editVectorStore}
    />,
  );

const savedPayload = () => mockUpdate.mock.calls[0][1];

describe("VectorStoreInfoView save payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockInfo.mockResolvedValue({ vector_store: serverRecord });
    mockCredentials.mockResolvedValue({ credentials: [{ credential_name: "bedrock-prod" }] });
    mockUpdate.mockResolvedValue({});
  });

  it("still saves when the server left the nullable name and description null", async () => {
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({
      vector_store: { ...serverRecord, vector_store_name: null, vector_store_description: null },
    });
    renderView(true);
    await screen.findByRole("button", { name: "Save Changes" });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: null,
      vector_store_description: null,
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("sends only the five editable keys and drops every server-only field", async () => {
    const user = userEvent.setup();
    renderView(true);

    const nameInput = await screen.findByDisplayValue("support-docs-store");
    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "renamed-store" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate.mock.calls[0][0]).toBe("sk-test");
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: "renamed-store",
      vector_store_description: "Docs for support",
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("sends the same five keys when editing is entered from the details view", async () => {
    const user = userEvent.setup();
    renderView(false);

    const editButtons = await screen.findAllByRole("button", { name: "Edit Vector Store" });
    await user.click(editButtons[0]);
    const descriptionInput = await screen.findByDisplayValue("Docs for support");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "new description" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: "support-docs-store",
      vector_store_description: "new description",
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("keeps the credential field out of the payload even after it is picked", async () => {
    const user = userEvent.setup();
    renderView(true);

    await screen.findByDisplayValue("support-docs-store");
    await user.click(screen.getAllByRole("combobox")[1]);
    const options = await screen.findAllByText("bedrock-prod");
    await user.click(options[options.length - 1]);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(Object.keys(savedPayload())).toStrictEqual([
      "vector_store_id",
      "custom_llm_provider",
      "vector_store_name",
      "vector_store_description",
      "vector_store_metadata",
    ]);
  });

  it("blocks the request and reports invalid metadata JSON instead of saving", async () => {
    const user = userEvent.setup();
    renderView(true);

    const metadataInput = await screen.findByPlaceholderText('{"key": "value"}');
    await user.clear(metadataInput);
    fireEvent.change(metadataInput, { target: { value: "not json" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockToast.fromError).toHaveBeenCalledWith("Invalid JSON in metadata field"));
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("keeps the required-field message that blocks saving without a vector store id", async () => {
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({ vector_store: { ...serverRecord, vector_store_id: "" } });
    renderView(true);

    await screen.findByDisplayValue("support-docs-store");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Please input a vector store ID")).toBeInTheDocument();
    expect(mockUpdate).not.toHaveBeenCalled();
  });
});
