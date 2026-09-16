import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
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
    renderWithProviders(
      <VectorStoreInfoView vectorStoreId="vs-1" onClose={vi.fn()} accessToken="sk-test" is_admin={true} />,
    );
    expect(await screen.findByText("Vector Store ID: vs-1")).toBeInTheDocument();
  });

  it("should show a not-found state with a working back button when the fetch fails instead of loading forever", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    mockVectorStoreInfoCall.mockRejectedValue(new Error("Vector store not found"));
    renderWithProviders(
      <VectorStoreInfoView vectorStoreId="vs-gone" onClose={onClose} accessToken="sk-test" is_admin={true} />,
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
    renderWithProviders(
      <VectorStoreInfoView vectorStoreId="vs-1" onClose={vi.fn()} accessToken="sk-test" is_admin={true} />,
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
    renderWithProviders(
      <VectorStoreInfoView vectorStoreId="vs-gone" onClose={vi.fn()} accessToken="sk-test" is_admin={true} />,
    );
    expect(await screen.findByText("Vector store not found")).toBeInTheDocument();
  });

  describe("URL state", () => {
    const storeRecord = {
      vector_store_id: "vs-1",
      vector_store_name: "support-docs-store",
      custom_llm_provider: "bedrock",
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
    };
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];
    const renderView = (searchParams: string, onUrlUpdate?: OnUrlUpdateFunction, isAdmin = true) =>
      renderWithProviders(
        <VectorStoreInfoView vectorStoreId="vs-1" onClose={vi.fn()} accessToken="sk-test" is_admin={isAdmin} />,
        { searchParams, onUrlUpdate },
      );

    beforeEach(() => {
      mockVectorStoreInfoCall.mockResolvedValue({ vector_store: storeRecord });
    });

    it("opens the tab named by detail_tab", async () => {
      renderView("?vector_store=vs-1&detail_tab=test");
      await screen.findByText("Vector Store ID: vs-1");
      expect(screen.getByRole("tab", { name: "Test Vector Store" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: "Details" })).toHaveAttribute("aria-selected", "false");
    });

    it("writes detail_tab when a tab is picked and drops it when returning to Details", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("?vector_store=vs-1", onUrlUpdate);
      await screen.findByText("Vector Store ID: vs-1");

      await user.click(screen.getByRole("tab", { name: "Test Vector Store" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("detail_tab")).toBe("test"));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("vector_store")).toBe("vs-1");

      await user.click(screen.getByRole("tab", { name: "Details" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("detail_tab")).toBe(false));
      expect(screen.getByRole("tab", { name: "Details" })).toHaveAttribute("aria-selected", "true");
    });

    it("falls back to Details for an unknown detail_tab and clears it from the URL", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      render(<VectorStoreInfoView vectorStoreId="vs-1" onClose={vi.fn()} accessToken="sk-test" is_admin={true} />, {
        wrapper: ({ children }: { children: ReactNode }) => (
          <NuqsTestingAdapter
            searchParams="?vector_store=vs-1&detail_tab=bogus"
            onUrlUpdate={onUrlUpdate}
            hasMemory
            resetUrlUpdateQueueOnMount={false}
          >
            {children}
          </NuqsTestingAdapter>
        ),
      });
      await screen.findByText("Vector Store ID: vs-1");
      expect(screen.getByRole("tab", { name: "Details" })).toHaveAttribute("aria-selected", "true");
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("detail_tab")).toBe(false));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("vector_store")).toBe("vs-1");
    });

    it("opens the edit form when edit=true is in the URL", async () => {
      renderView("?vector_store=vs-1&edit=true");
      expect(await screen.findByRole("button", { name: "Save Changes" })).toBeInTheDocument();
    });

    it("writes edit=true when editing starts and clears it on cancel", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("?vector_store=vs-1", onUrlUpdate);
      await screen.findByText("Vector Store ID: vs-1");

      await user.click(screen.getAllByRole("button", { name: "Edit Vector Store" })[0]);
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("edit")).toBe("true"));
      expect(await screen.findByRole("button", { name: "Save Changes" })).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Cancel" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("edit")).toBe(false));
      expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    });

    it("opens the edit form from edit=true for a non-admin, as the table's Edit action does", async () => {
      renderView("?vector_store=vs-1&edit=true", undefined, false);
      expect(await screen.findByRole("button", { name: "Save Changes" })).toBeInTheDocument();
    });

    it("hides the Edit Vector Store button from a non-admin", async () => {
      renderView("?vector_store=vs-1", undefined, false);
      await screen.findByText("Vector Store ID: vs-1");
      expect(screen.queryByRole("button", { name: "Edit Vector Store" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    });
  });
});
