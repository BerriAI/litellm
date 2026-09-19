import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { useQueryState } from "nuqs";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
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
  default: ({
    isLoading,
    onView,
    onEdit,
  }: {
    isLoading?: boolean;
    onView: (vectorStoreId: string) => void;
    onEdit: (vectorStoreId: string) => void;
  }) => (
    <div data-testid="vector-store-table">
      {isLoading ? "table-loading" : "table-loaded"}
      <button type="button" onClick={() => onView("vs-1")}>
        View vs-1
      </button>
      <button type="button" onClick={() => onEdit("vs-1")}>
        Edit vs-1
      </button>
    </div>
  ),
}));

vi.mock("./VectorStoreForm", () => ({ __esModule: true, default: () => null }));
vi.mock("./vector_store_info", () => ({
  __esModule: true,
  default: ({ vectorStoreId, onClose }: { vectorStoreId: string; onClose: () => void }) => (
    <div>
      <div data-testid="vector-store-info-view">{vectorStoreId}</div>
      <button type="button" onClick={onClose}>
        Close info
      </button>
    </div>
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
    renderWithProviders(<VectorStoreManagement accessToken={null} userID={null} userRole={null} isViewOnly={false} />);
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
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
    );
    await openManageTab(user);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ data: [] });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test");
  });
});

describe("VectorStoreManagement create flow visibility", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorStoreListCall.mockResolvedValue({ data: [] });
    mockCredentialListCall.mockResolvedValue({ credentials: [] });
  });

  it.each([
    { label: "Internal User", userRole: "Internal User", isViewOnly: false },
    { label: "Internal Viewer", userRole: "Internal Viewer", isViewOnly: true },
    { label: "proxy_admin_viewer session (userRole Admin, isViewOnly)", userRole: "Admin", isViewOnly: true },
    { label: "Org Admin", userRole: "Org Admin", isViewOnly: false },
  ])(
    "should hide the Create Vector Store tab and button and skip /credentials for $label",
    async ({ userRole, isViewOnly }) => {
      renderWithProviders(
        <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole={userRole} isViewOnly={isViewOnly} />,
      );
      await waitFor(() => expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test"));
      expect(screen.queryByRole("tab", { name: "Create Vector Store" })).not.toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Manage Vector Stores" })).toHaveAttribute("aria-selected", "true");
      expect(await screen.findByText("table-loaded")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "+ Add Vector Store" })).not.toBeInTheDocument();
      expect(mockCredentialListCall).not.toHaveBeenCalled();
    },
  );

  it("should keep the Create Vector Store tab and button and fetch /credentials for a proxy admin", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
    );
    await waitFor(() => expect(mockCredentialListCall).toHaveBeenCalledWith("sk-test"));
    expect(screen.getByRole("tab", { name: "Create Vector Store" })).toHaveAttribute("aria-selected", "true");
    await openManageTab(user);
    expect(screen.getByRole("button", { name: "+ Add Vector Store" })).toBeInTheDocument();
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
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
    );
    await user.click(screen.getByRole("tab", { name: "Indexes" }));
    expect(await screen.findByText("support-docs-index")).toBeInTheDocument();
    expect(screen.getByText("support-docs-store")).toBeInTheDocument();
    expect(mockIndexesListCall).toHaveBeenCalledWith("sk-test");
  });

  it("should not render the Indexes tab for an Admin Viewer", async () => {
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin Viewer" isViewOnly={true} />,
    );
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
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
    );
    await user.click(screen.getByRole("tab", { name: "Indexes" }));
    await user.click(await screen.findByRole("button", { name: "support-docs-store" }));
    expect(await screen.findByTestId("vector-store-info-view")).toHaveTextContent("vs-1");
    expect(screen.queryByText("Vector Store Management")).not.toBeInTheDocument();
  });

  it("should link to the feature docs and a GitHub issue for unsupported providers on the Indexes tab", async () => {
    const user = userEvent.setup();
    mockIndexesListCall.mockResolvedValue({ object: "list", data: [] });
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
    );
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
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
    );
    await waitFor(() => expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test"));
    expect(screen.getByRole("tab", { name: "Indexes" })).toBeInTheDocument();
    expect(mockIndexesListCall).not.toHaveBeenCalled();
  });
});

describe("VectorStoreManagement URL state", () => {
  const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
    onUrlUpdate.mock.calls.at(-1)?.[0];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorStoreListCall.mockResolvedValue({ data: [] });
    mockCredentialListCall.mockResolvedValue({ credentials: [] });
    mockIndexesListCall.mockResolvedValue({ object: "list", data: [] });
  });

  it("opens the tab named in the URL for a proxy admin", async () => {
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
      { searchParams: "?tab=indexes" },
    );
    expect(screen.getByRole("tab", { name: "Indexes" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(mockIndexesListCall).toHaveBeenCalledWith("sk-test"));
  });

  it("writes the picked tab to the URL and clears it when returning to the default tab", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
      { onUrlUpdate },
    );

    await user.click(screen.getByRole("tab", { name: "Test Vector Store" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("test"));
    expect(screen.getByRole("tab", { name: "Test Vector Store" })).toHaveAttribute("aria-selected", "true");

    await user.click(screen.getByRole("tab", { name: "Create Vector Store" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
    expect(screen.getByRole("tab", { name: "Create Vector Store" })).toHaveAttribute("aria-selected", "true");
  });

  it.each([
    { tab: "create", userRole: "Internal User", isViewOnly: false },
    { tab: "indexes", userRole: "Admin Viewer", isViewOnly: true },
  ])(
    "falls back to Manage and clears tab=$tab for a $userRole who cannot see it",
    async ({ tab, userRole, isViewOnly }) => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      render(
        <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole={userRole} isViewOnly={isViewOnly} />,
        {
          wrapper: ({ children }: { children: ReactNode }) => (
            <NuqsTestingAdapter
              searchParams={`?tab=${tab}`}
              onUrlUpdate={onUrlUpdate}
              hasMemory
              resetUrlUpdateQueueOnMount={false}
            >
              {children}
            </NuqsTestingAdapter>
          ),
        },
      );

      expect(screen.getByRole("tab", { name: "Manage Vector Stores" })).toHaveAttribute("aria-selected", "true");
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
      expect(mockIndexesListCall).not.toHaveBeenCalled();
    },
  );

  it("renders the info view for the vector store in the URL", async () => {
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
      { searchParams: "?vector_store=vs-1" },
    );
    expect(screen.getByTestId("vector-store-info-view")).toHaveTextContent("vs-1");
    expect(screen.queryByText("Vector Store Management")).not.toBeInTheDocument();
  });

  it("pushes vector_store when an index's vector store is opened", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
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
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
      { searchParams: "?tab=indexes", onUrlUpdate },
    );

    await user.click(await screen.findByRole("button", { name: "support-docs-store" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("vector_store")).toBe("vs-1"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("indexes");
  });

  it("clears every detail key when the info view closes and returns to the tab it came from", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
      { searchParams: "?tab=manage&vector_store=vs-1&edit=true&detail_tab=test", onUrlUpdate },
    );

    await user.click(screen.getByRole("button", { name: "Close info" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("vector_store")).toBe(false));
    const params = lastUrlUpdate(onUrlUpdate)?.searchParams;
    expect(params?.has("edit")).toBe(false);
    expect(params?.has("detail_tab")).toBe(false);
    expect(params?.get("tab")).toBe("manage");
    expect(await screen.findByRole("tab", { name: "Manage Vector Stores" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(mockVectorStoreListCall).toHaveBeenCalledTimes(1));
  });

  it("reloads the vector store list when the detail closes through the URL alone, as browser Back does", async () => {
    const user = userEvent.setup();
    const ClearVectorStoreParam = () => {
      const [, setVectorStore] = useQueryState("vector_store");
      return (
        <button type="button" onClick={() => void setVectorStore(null)}>
          Browser back
        </button>
      );
    };
    renderWithProviders(
      <>
        <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />
        <ClearVectorStoreParam />
      </>,
      { searchParams: "?tab=manage&vector_store=vs-1" },
    );
    expect(screen.getByTestId("vector-store-info-view")).toHaveTextContent("vs-1");
    expect(mockVectorStoreListCall).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Browser back" }));

    await waitFor(() => expect(mockVectorStoreListCall).toHaveBeenCalledTimes(1));
    expect(mockCredentialListCall).toHaveBeenCalledWith("sk-test");
    expect(await screen.findByTestId("vector-store-table")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "View vs-1" }));
    expect(await screen.findByTestId("vector-store-info-view")).toHaveTextContent("vs-1");
    await user.click(screen.getByRole("button", { name: "Browser back" }));

    await waitFor(() => expect(mockVectorStoreListCall).toHaveBeenCalledTimes(2));
  });

  it("pushes vector_store with edit=true from the table's Edit action and drops a stale detail_tab", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
      { searchParams: "?tab=manage&detail_tab=test", onUrlUpdate },
    );

    await user.click(await screen.findByRole("button", { name: "Edit vs-1" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("vector_store")).toBe("vs-1"));
    const update = lastUrlUpdate(onUrlUpdate);
    expect(update?.searchParams.get("edit")).toBe("true");
    expect(update?.searchParams.has("detail_tab")).toBe(false);
    expect(update?.searchParams.get("tab")).toBe("manage");
    expect(update?.options.history).toBe("push");
    expect(screen.getByTestId("vector-store-info-view")).toHaveTextContent("vs-1");
  });

  it("pushes vector_store without edit from the table's View action even when edit=true is left over", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(
      <VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" isViewOnly={false} />,
      { searchParams: "?tab=manage&edit=true&detail_tab=test", onUrlUpdate },
    );

    await user.click(await screen.findByRole("button", { name: "View vs-1" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("vector_store")).toBe("vs-1"));
    const update = lastUrlUpdate(onUrlUpdate);
    expect(update?.searchParams.has("edit")).toBe(false);
    expect(update?.searchParams.has("detail_tab")).toBe(false);
    expect(update?.options.history).toBe("push");
  });
});
