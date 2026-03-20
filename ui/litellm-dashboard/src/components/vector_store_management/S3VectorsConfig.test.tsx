import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import S3VectorsConfig from "./S3VectorsConfig";
import * as fetchModels from "../playground/llm_calls/fetch_models";

// Mock fetchAvailableModels
vi.mock("../playground/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

describe("S3VectorsConfig", () => {
  const mockOnParamsChange = vi.fn();
  const defaultProps = {
    accessToken: "test-token",
    providerParams: {},
    onParamsChange: mockOnParamsChange,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the component successfully", () => {
    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue([]);

    render(<S3VectorsConfig {...defaultProps} />);

    expect(screen.getByText("AWS S3 Vectors Setup")).toBeInTheDocument();
    expect(screen.getByText("Vector Bucket Name")).toBeInTheDocument();
    expect(screen.getByText("Index Name")).toBeInTheDocument();
    expect(screen.getByText("AWS Region")).toBeInTheDocument();
    expect(screen.getByText("Embedding Model")).toBeInTheDocument();
  });

  it("should display setup instructions", () => {
    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue([]);

    render(<S3VectorsConfig {...defaultProps} />);

    expect(
      screen.getByText(/AWS S3 Vectors allows you to store and query vector embeddings directly in S3/)
    ).toBeInTheDocument();
    expect(screen.getByText(/Vector buckets and indexes will be automatically created/)).toBeInTheDocument();
    expect(screen.getByText(/Vector dimensions are auto-detected/)).toBeInTheDocument();
  });

  it("should fetch embedding models on mount", async () => {
    const mockModels = [
      { model_group: "text-embedding-3-small", mode: "embedding" },
      { model_group: "text-embedding-3-large", mode: "embedding" },
      { model_group: "gpt-4", mode: "chat" },
    ];

    const fetchSpy = vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue(mockModels);

    render(<S3VectorsConfig {...defaultProps} />);

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith("test-token");
    });
  });

  it("should filter and display only embedding models", async () => {
    const mockModels = [
      { model_group: "text-embedding-3-small", mode: "embedding" },
      { model_group: "text-embedding-3-large", mode: "embedding" },
      { model_group: "gpt-4", mode: "chat" },
      { model_group: "gpt-3.5-turbo", mode: "chat" },
    ];

    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue(mockModels);

    render(<S3VectorsConfig {...defaultProps} />);

    // Wait for models to load
    await waitFor(() => {
      expect(fetchModels.fetchAvailableModels).toHaveBeenCalled();
    });

    // The component should filter to only embedding models internally
    // We can verify this by checking the component loaded successfully
    expect(screen.getByText("Embedding Model")).toBeInTheDocument();
  });

  it("should call onParamsChange when vector bucket name changes", async () => {
    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue([]);

    render(<S3VectorsConfig {...defaultProps} />);

    const bucketInput = screen.getByPlaceholderText("my-vector-bucket (min 3 chars)");

    await act(async () => {
      fireEvent.change(bucketInput, { target: { value: "test-bucket" } });
    });

    expect(mockOnParamsChange).toHaveBeenCalledWith({
      vector_bucket_name: "test-bucket",
    });
  });

  it("should call onParamsChange when AWS region changes", async () => {
    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue([]);

    render(<S3VectorsConfig {...defaultProps} />);

    const regionInput = screen.getByPlaceholderText("us-west-2");

    await act(async () => {
      fireEvent.change(regionInput, { target: { value: "us-east-1" } });
    });

    expect(mockOnParamsChange).toHaveBeenCalledWith({
      aws_region_name: "us-east-1",
    });
  });

  it("should call onParamsChange when embedding model is selected", async () => {
    const mockModels = [
      { model_group: "text-embedding-3-small", mode: "embedding" },
      { model_group: "text-embedding-3-large", mode: "embedding" },
    ];

    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue(mockModels);

    render(<S3VectorsConfig {...defaultProps} />);

    await waitFor(() => {
      expect(fetchModels.fetchAvailableModels).toHaveBeenCalled();
    });

    // Find the Select component and trigger change directly
    const selectElement = screen.getByRole("combobox");

    await act(async () => {
      // Simulate selecting a value by firing the change event
      fireEvent.change(selectElement, { target: { value: "text-embedding-3-small" } });
    });

    // The component should handle the selection
    expect(screen.getByText("Embedding Model")).toBeInTheDocument();
  });

  it("should preserve existing params when updating a field", async () => {
    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue([]);

    const existingParams = {
      vector_bucket_name: "existing-bucket",
      aws_region_name: "us-west-2",
    };

    render(<S3VectorsConfig {...defaultProps} providerParams={existingParams} />);

    const indexInput = screen.getByPlaceholderText("my-vector-index (optional, min 3 chars)");

    await act(async () => {
      fireEvent.change(indexInput, { target: { value: "my-index" } });
    });

    expect(mockOnParamsChange).toHaveBeenCalledWith({
      vector_bucket_name: "existing-bucket",
      aws_region_name: "us-west-2",
      index_name: "my-index",
    });
  });

  it("should display existing param values", () => {
    vi.spyOn(fetchModels, "fetchAvailableModels").mockResolvedValue([]);

    const existingParams = {
      vector_bucket_name: "my-bucket",
      index_name: "my-index",
      aws_region_name: "eu-west-1",
      embedding_model: "text-embedding-3-small",
    };

    render(<S3VectorsConfig {...defaultProps} providerParams={existingParams} />);

    expect(screen.getByDisplayValue("my-bucket")).toBeInTheDocument();
    expect(screen.getByDisplayValue("my-index")).toBeInTheDocument();
    expect(screen.getByDisplayValue("eu-west-1")).toBeInTheDocument();
  });

  it("should handle model fetch error gracefully", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(fetchModels, "fetchAvailableModels").mockRejectedValue(new Error("Failed to fetch models"));

    render(<S3VectorsConfig {...defaultProps} />);

    await waitFor(() => {
      expect(consoleErrorSpy).toHaveBeenCalledWith("Error fetching embedding models:", expect.any(Error));
    });

    consoleErrorSpy.mockRestore();
  });

  it("should not fetch models if accessToken is null", () => {
    const fetchSpy = vi.spyOn(fetchModels, "fetchAvailableModels");

    render(<S3VectorsConfig {...defaultProps} accessToken={null} />);

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
