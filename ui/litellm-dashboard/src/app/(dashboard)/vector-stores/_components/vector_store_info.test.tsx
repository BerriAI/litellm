import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { credentialListCall, vectorStoreInfoCall } from "@/components/networking";

import VectorStoreInfoView from "./vector_store_info";

vi.mock("@/components/networking", () => ({
  vectorStoreInfoCall: vi.fn(),
  vectorStoreUpdateCall: vi.fn(),
  credentialListCall: vi.fn(),
}));

vi.mock("./VectorStoreTester", async () => {
  const { useState } = await import("react");
  const VectorStoreTesterStub = () => {
    const [searchesRun, setSearchesRun] = useState(0);
    return (
      <div>
        <button type="button" onClick={() => setSearchesRun((count) => count + 1)}>
          Run search
        </button>
        <p>Searches run: {searchesRun}</p>
      </div>
    );
  };
  return { __esModule: true, default: VectorStoreTesterStub };
});

const mockVectorStoreInfoCall = vi.mocked(vectorStoreInfoCall);
const mockCredentialListCall = vi.mocked(credentialListCall);

describe("VectorStoreInfoView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCredentialListCall.mockResolvedValue({ credentials: [] });
  });

  it("should render the store details once the fetch resolves", async () => {
    mockVectorStoreInfoCall.mockResolvedValue({
      vector_store: {
        vector_store_id: "vs-1",
        vector_store_name: "support-docs-store",
        custom_llm_provider: "bedrock",
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      },
    });
    render(
      <VectorStoreInfoView
        vectorStoreId="vs-1"
        onClose={vi.fn()}
        accessToken="sk-test"
        is_admin={true}
        editVectorStore={false}
      />,
    );
    expect(await screen.findByText("Vector Store ID: vs-1")).toBeInTheDocument();
  });

  it("should show a not-found state with a working back button when the fetch fails instead of loading forever", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    mockVectorStoreInfoCall.mockRejectedValue(new Error("Vector store not found"));
    render(
      <VectorStoreInfoView
        vectorStoreId="vs-gone"
        onClose={onClose}
        accessToken="sk-test"
        is_admin={true}
        editVectorStore={false}
      />,
    );
    expect(await screen.findByText("Vector store not found")).toBeInTheDocument();
    expect(screen.getByText(/vs-gone could not be loaded/)).toBeInTheDocument();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Back to Vector Stores/ }));
    expect(onClose).toHaveBeenCalled();
  });

  it("keeps the test panel's search state when switching to Details and back", async () => {
    const user = userEvent.setup();
    mockVectorStoreInfoCall.mockResolvedValue({
      vector_store: {
        vector_store_id: "vs-1",
        vector_store_name: "support-docs-store",
        custom_llm_provider: "bedrock",
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      },
    });
    render(
      <VectorStoreInfoView
        vectorStoreId="vs-1"
        onClose={vi.fn()}
        accessToken="sk-test"
        is_admin={true}
        editVectorStore={false}
      />,
    );
    expect(await screen.findByText("Vector Store ID: vs-1")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Test Vector Store" }));
    await user.click(screen.getByRole("button", { name: "Run search" }));
    expect(screen.getByText("Searches run: 1")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Details" }));
    await user.click(screen.getByRole("tab", { name: "Test Vector Store" }));
    expect(screen.getByText("Searches run: 1")).toBeInTheDocument();
  });

  it("should show the not-found state when the fetch resolves without a vector store", async () => {
    mockVectorStoreInfoCall.mockResolvedValue({ vector_store: null });
    render(
      <VectorStoreInfoView
        vectorStoreId="vs-gone"
        onClose={vi.fn()}
        accessToken="sk-test"
        is_admin={true}
        editVectorStore={false}
      />,
    );
    expect(await screen.findByText("Vector store not found")).toBeInTheDocument();
  });
});
