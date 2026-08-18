import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { credentialListCall, indexesListCall, vectorStoreListCall } from "@/components/networking";

import VectorStoreManagement from "./index";

vi.mock("@/components/networking", () => ({
  vectorStoreListCall: vi.fn(),
  vectorStoreDeleteCall: vi.fn(),
  credentialListCall: vi.fn(),
  indexesListCall: vi.fn(),
}));

vi.mock("./VectorStoreTable", () => ({
  __esModule: true,
  default: ({ isLoading }: { isLoading?: boolean }) => (
    <div data-testid="vector-store-table">{isLoading ? "table-loading" : "table-loaded"}</div>
  ),
}));

vi.mock("./VectorStoreForm", () => ({ __esModule: true, default: () => null }));
vi.mock("./vector_store_info", () => ({
  __esModule: true,
  default: ({ vectorStoreId }: { vectorStoreId: string }) => (
    <div data-testid="vector-store-info-view">{vectorStoreId}</div>
  ),
}));
vi.mock("./CreateVectorStore", () => ({ __esModule: true, default: () => null }));
vi.mock("./TestVectorStoreTab", () => ({ __esModule: true, default: () => null }));

const mockVectorStoreListCall = vi.mocked(vectorStoreListCall);
const mockCredentialListCall = vi.mocked(credentialListCall);
const mockIndexesListCall = vi.mocked(indexesListCall);

const openManageTab = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("tab", { name: "Manage Vector Stores" }));
};

describe("VectorStoreManagement loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    const user = userEvent.setup();
    render(<VectorStoreManagement accessToken={null} userID={null} userRole={null} />);
    await openManageTab(user);
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockVectorStoreListCall).not.toHaveBeenCalled();
  });

  it("should show the loading state until the vector store fetch settles", async () => {
    const user = userEvent.setup();
    let resolveFetch: (value: { data: never[] }) => void = () => {};
    mockVectorStoreListCall.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await openManageTab(user);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ data: [] });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test");
  });
});

describe("VectorStoreManagement Indexes tab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorStoreListCall.mockResolvedValue({ data: [] });
    mockCredentialListCall.mockResolvedValue({ credentials: [] });
  });

  it("should render fetched indexes for a proxy admin after the Indexes tab is clicked", async () => {
    const user = userEvent.setup();
    mockIndexesListCall.mockResolvedValue({
      object: "list",
      data: [
        {
          id: "idx-1",
          index_name: "support-docs-index",
          litellm_params: { vector_store_index: "pinecone-support-docs", vector_store_name: "support-docs-store" },
        },
      ],
    });
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await user.click(screen.getByRole("tab", { name: "Indexes" }));
    expect(await screen.findByText("support-docs-index")).toBeInTheDocument();
    expect(screen.getByText("support-docs-store")).toBeInTheDocument();
    expect(mockIndexesListCall).toHaveBeenCalledWith("sk-test");
  });

  it("should not render the Indexes tab for an Admin Viewer", async () => {
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin Viewer" />);
    await waitFor(() => expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test"));
    expect(screen.getByRole("tab", { name: "Manage Vector Stores" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Indexes" })).not.toBeInTheDocument();
  });

  it("should swap to the vector store info view when an index's vector store name is clicked", async () => {
    const user = userEvent.setup();
    mockVectorStoreListCall.mockResolvedValue({
      data: [
        {
          vector_store_id: "vs-1",
          vector_store_name: "support-docs-store",
          custom_llm_provider: "bedrock",
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ],
    });
    mockIndexesListCall.mockResolvedValue({
      object: "list",
      data: [
        {
          id: "idx-1",
          index_name: "support-docs-index",
          litellm_params: { vector_store_index: "pinecone-support-docs", vector_store_name: "support-docs-store" },
        },
      ],
    });
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await user.click(screen.getByRole("tab", { name: "Indexes" }));
    await user.click(await screen.findByRole("button", { name: "support-docs-store" }));
    expect(await screen.findByTestId("vector-store-info-view")).toHaveTextContent("vs-1");
    expect(screen.queryByText("Vector Store Management")).not.toBeInTheDocument();
  });

  it("should link to the feature docs and a GitHub issue for unsupported providers on the Indexes tab", async () => {
    const user = userEvent.setup();
    mockIndexesListCall.mockResolvedValue({ object: "list", data: [] });
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await user.click(screen.getByRole("tab", { name: "Indexes" }));
    expect(screen.getByRole("link", { name: "vector store index docs" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/providers/azure_ai/azure_ai_vector_stores_passthrough",
    );
    expect(screen.getByRole("link", { name: "file a GitHub issue" })).toHaveAttribute(
      "href",
      "https://github.com/BerriAI/litellm/issues",
    );
    expect(screen.getByText(/supported for Azure AI Search and Milvus today/)).toBeInTheDocument();
  });

  it("should not call indexesListCall until the Indexes tab is clicked", async () => {
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await waitFor(() => expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test"));
    expect(screen.getByRole("tab", { name: "Indexes" })).toBeInTheDocument();
    expect(mockIndexesListCall).not.toHaveBeenCalled();
  });
});
