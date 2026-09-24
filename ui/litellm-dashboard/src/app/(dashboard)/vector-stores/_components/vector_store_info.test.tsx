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

  it("should render a config-defined store read-only for an admin, even when opened in edit mode", async () => {
    mockVectorStoreInfoCall.mockResolvedValue({
      vector_store: {
        vector_store_id: "vs-config",
        vector_store_name: "config-store",
        custom_llm_provider: "openai",
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
        is_config: true,
      },
    });
    render(
      <VectorStoreInfoView
        vectorStoreId="vs-config"
        onClose={vi.fn()}
        accessToken="sk-test"
        is_admin={true}
        editVectorStore={true}
      />,
    );
    expect(await screen.findByText("Vector Store ID: vs-config")).toBeInTheDocument();
    expect(screen.getByText("Read only: defined in the config file")).toBeInTheDocument();
    expect(screen.getByText("Config")).toBeInTheDocument();
    expect(screen.queryByText("DB")).not.toBeInTheDocument();
    expect(screen.getByText("Vector Store Details")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit Vector Store" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Save/ })).not.toBeInTheDocument();
  });

  it("should still offer editing for a database-backed store", async () => {
    mockVectorStoreInfoCall.mockResolvedValue({
      vector_store: {
        vector_store_id: "vs-db",
        vector_store_name: "db-store",
        custom_llm_provider: "openai",
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
        is_config: false,
      },
    });
    render(
      <VectorStoreInfoView
        vectorStoreId="vs-db"
        onClose={vi.fn()}
        accessToken="sk-test"
        is_admin={true}
        editVectorStore={false}
      />,
    );
    expect(await screen.findByText("Vector Store ID: vs-db")).toBeInTheDocument();
    expect(screen.queryByText("Read only: defined in the config file")).not.toBeInTheDocument();
    expect(screen.getByText("DB")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Edit Vector Store" }).length).toBeGreaterThan(0);
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
