import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import TestVectorStoreTab from "./TestVectorStoreTab";
import { VectorStore } from "./types";

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
    render(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    expect(screen.getByText("Select Vector Store")).toBeInTheDocument();
    expect(screen.getByText("Choose a vector store to test search queries against")).toBeInTheDocument();
  });

  it("should show message when no access token", () => {
    render(<TestVectorStoreTab accessToken={null} vectorStores={mockVectorStores} />);

    expect(screen.getByText("Access token is required to test vector stores.")).toBeInTheDocument();
  });

  it("should show message when no vector stores available", () => {
    render(<TestVectorStoreTab accessToken="test-token" vectorStores={[]} />);

    expect(screen.getByText("No vector stores available. Create one first to test it.")).toBeInTheDocument();
  });

  it("should render VectorStoreTester with first vector store by default", () => {
    render(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    expect(screen.getByTestId("vector-store-tester")).toBeInTheDocument();
    expect(screen.getByTestId("tester-vector-store-id")).toHaveTextContent("vs_123");
    expect(screen.getByTestId("tester-access-token")).toHaveTextContent("test-token");
  });

  it("should update VectorStoreTester when selecting different vector store", () => {
    render(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    // Find the select component
    const selectElement = screen.getByRole("combobox");

    // Change selection
    fireEvent.mouseDown(selectElement);

    // Wait for options to appear and click the second one
    const option2 = screen.getByText("Test Store 2");
    fireEvent.click(option2);

    // Verify the tester component updated
    expect(screen.getByTestId("tester-vector-store-id")).toHaveTextContent("vs_456");
  });

  it("should display vector store names in select options", () => {
    render(<TestVectorStoreTab accessToken="test-token" vectorStores={mockVectorStores} />);

    const selectElement = screen.getByRole("combobox");
    fireEvent.mouseDown(selectElement);

    // Use getAllByText since the selected value also shows the name
    expect(screen.getAllByText("Test Store 1").length).toBeGreaterThan(0);
    expect(screen.getByText("Test Store 2")).toBeInTheDocument();
  });
});
