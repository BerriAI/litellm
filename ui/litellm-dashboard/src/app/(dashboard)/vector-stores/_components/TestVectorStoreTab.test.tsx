import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { renderWithProviders } from "@/../tests/test-utils";
import { describe, it, expect, vi } from "vitest";
import TestVectorStoreTab from "./TestVectorStoreTab";
import { VectorStore } from "@/components/vector_store_management/types";

// Mock VectorStoreTester component
vi.mock("./VectorStoreTester", () => ({
  VectorStoreTester: ({ vectorStoreId, accessToken }: { vectorStoreId: string; accessToken: string }) => (
    <div data-testid="vector-store-tester">
      <div data-testid="tester-vector-store-id">{vectorStoreId}</div>
      <div data-testid="tester-access-token">{accessToken}</div>
    </div>
  ),
}));

const mockVectorStores: VectorStore[] = [
  {
    vector_store_id: "vs_123",
    custom_llm_provider: "openai",
    vector_store_name: "Test Store 1",
    vector_store_description: "Description 1",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  },
  {
    vector_store_id: "vs_456",
    custom_llm_provider: "bedrock",
    vector_store_name: "Test Store 2",
    vector_store_description: "Description 2",
    created_at: "2024-01-02T00:00:00Z",
    updated_at: "2024-01-02T00:00:00Z",
  },
];

describe("TestVectorStoreTab", () => {
  it("should render the component successfully", () => {
    renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    expect(screen.getByText("Select Vector Store")).toBeInTheDocument();
    expect(screen.getByText("Choose a vector store to test search queries against")).toBeInTheDocument();
  });

  it("should show message when no access token", () => {
    renderWithProviders(<TestVectorStoreTab accessToken={null} vectorStores={mockVectorStores} />);

    expect(screen.getByText("Access token is required to test vector stores.")).toBeInTheDocument();
  });

  it("should show message when no vector stores available", () => {
    renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={[]} />);

    expect(screen.getByText("No vector stores available. Create one first to test it.")).toBeInTheDocument();
  });

  it("should render VectorStoreTester with first vector store by default", () => {
    renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    expect(screen.getByTestId("vector-store-tester")).toBeInTheDocument();
    expect(screen.getByTestId("tester-vector-store-id")).toHaveTextContent("vs_123");
    expect(screen.getByTestId("tester-access-token")).toHaveTextContent("test-token");
  });

  it("should update VectorStoreTester when selecting different vector store", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Test Store 2"));

    expect(screen.getByTestId("tester-vector-store-id")).toHaveTextContent("vs_456");
  });

  it("should display vector store names in select options", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    await user.click(screen.getByRole("combobox"));

    // The selected store's name may also render in the trigger, so only require at least one match.
    expect((await screen.findAllByText("Test Store 1")).length).toBeGreaterThan(0);
    expect(screen.getByText("Test Store 2")).toBeInTheDocument();
  });

  describe("URL state", () => {
    it("tests the vector store named in the URL", () => {
      renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />, {
        searchParams: "?test_vector_store=vs_456",
      });

      expect(screen.getByTestId("tester-vector-store-id")).toHaveTextContent("vs_456");
      expect(screen.getByRole("combobox")).toHaveValue("Test Store 2");
    });

    it("writes the picked vector store id to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />, {
        onUrlUpdate,
      });

      await user.click(screen.getByRole("combobox"));
      await user.click(await screen.findByText("Test Store 2"));

      await waitFor(() =>
        expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("test_vector_store")).toBe("vs_456"),
      );
    });

    it("falls back to the first vector store when the URL names one that no longer exists", () => {
      renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />, {
        searchParams: "?test_vector_store=vs_deleted",
      });

      expect(screen.getByTestId("tester-vector-store-id")).toHaveTextContent("vs_123");
    });

    it("follows the vector store list when the URL store arrives after the first render", () => {
      const { rerender } = renderWithProviders(<TestVectorStoreTab accessToken="test-token" vectorStores={[]} />, {
        searchParams: "?test_vector_store=vs_456",
      });

      rerender(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

      expect(screen.getByTestId("tester-vector-store-id")).toHaveTextContent("vs_456");
    });
  });
});
