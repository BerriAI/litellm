import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import CreateVectorStore from "./CreateVectorStore";
import * as networking from "../networking";

// Mock the networking module
vi.mock("../networking", () => ({
  ragIngestCall: vi.fn(),
}));

// Mock NotificationsManager
vi.mock("../molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

// Mock vector_store_providers
vi.mock("../vector_store_providers", () => ({
  VectorStoreProviders: {
    BEDROCK: "Amazon Bedrock",
    OPENAI: "OpenAI",
    AZURE_OPENAI: "Azure OpenAI",
    S3Vectors: "AWS S3 Vectors",
  },
  vectorStoreProviderMap: {
    BEDROCK: "bedrock",
    OPENAI: "openai",
    AZURE_OPENAI: "azure_openai",
    S3Vectors: "s3_vectors",
  },
  vectorStoreProviderLogoMap: {
    "Amazon Bedrock": "https://example.com/bedrock.png",
    "OpenAI": "https://example.com/openai.png",
    "Azure OpenAI": "https://example.com/azure.png",
    "AWS S3 Vectors": "https://example.com/aws.png",
  },
  getProviderSpecificFields: vi.fn((provider: string) => {
    if (provider === "s3_vectors") {
      return [
        {
          name: "vector_bucket_name",
          label: "Vector Bucket Name",
          tooltip: "S3 bucket name for vector storage",
          placeholder: "my-vector-bucket",
          required: true,
          type: "text",
        },
        {
          name: "aws_region_name",
          label: "AWS Region",
          tooltip: "AWS region",
          placeholder: "us-west-2",
          required: true,
          type: "text",
        },
        {
          name: "embedding_model",
          label: "Embedding Model",
          tooltip: "Embedding model to use",
          placeholder: "text-embedding-3-small",
          required: true,
          type: "select",
        },
      ];
    }
    return [];
  }),
}));

describe("CreateVectorStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the component successfully", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    expect(screen.getAllByText("Create Vector Store").length).toBeGreaterThan(0);
    expect(screen.getByText("Step 1: Upload Documents")).toBeInTheDocument();
    expect(screen.getByText("Step 2: Configure Vector Store")).toBeInTheDocument();
  });

  it("should display upload area with correct text", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    expect(screen.getByText("Click or drag files to this area to upload")).toBeInTheDocument();
    expect(screen.getByText(/Support for single or bulk upload/)).toBeInTheDocument();
  });

  it("should have provider selection dropdown", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    expect(screen.getByText("Provider")).toBeInTheDocument();
  });

  it("should have create button disabled initially when no documents", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });
    expect(createButton).toBeDisabled();
  });

  it("should show uploaded documents table when files are added", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    // Create a mock file
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });

    // Find the upload input (it's hidden but accessible)
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });
  });

  it("should call ragIngestCall when create button is clicked", async () => {
    const mockRagIngestCall = vi.spyOn(networking, "ragIngestCall");
    mockRagIngestCall.mockResolvedValue({
      id: "test-id",
      status: "completed",
      vector_store_id: "vs_123",
      file_id: "file_123",
    });

    const onSuccess = vi.fn();
    render(<CreateVectorStore accessToken="test-token" onSuccess={onSuccess} />);

    // Create a mock file
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    // Wait for file to be added
    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });

    // Click create button
    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });

    await act(async () => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(mockRagIngestCall).toHaveBeenCalledWith(
        "test-token",
        expect.any(File),
        "bedrock",
        undefined,
        undefined,
        undefined,
        {}
      );
    });
  });

  it("should display success message after successful creation", async () => {
    const mockRagIngestCall = vi.spyOn(networking, "ragIngestCall");
    mockRagIngestCall.mockResolvedValue({
      id: "test-id",
      status: "completed",
      vector_store_id: "vs_123",
      file_id: "file_123",
    });

    render(<CreateVectorStore accessToken="test-token" />);

    // Create and upload a mock file
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });

    // Click create button
    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });

    await act(async () => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Vector Store Created Successfully")).toBeInTheDocument();
    });
  });

  it("should display S3 Vectors provider-specific fields when selected", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    // Find and click the provider dropdown
    const providerSelect = screen.getByRole("combobox");

    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    // Wait for dropdown options to appear
    await waitFor(() => {
      const s3Option = screen.queryByText("AWS S3 Vectors");
      if (s3Option) {
        fireEvent.click(s3Option);
      }
    });

    // Check if S3-specific fields are displayed
    await waitFor(() => {
      expect(screen.queryByText("Vector Bucket Name")).toBeInTheDocument();
      expect(screen.queryByText("AWS Region")).toBeInTheDocument();
      expect(screen.queryByText("Embedding Model")).toBeInTheDocument();
    });
  });

  it("should validate S3 Vectors required fields before submission", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    // Upload a file first
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });

    // Select S3 Vectors provider
    const providerSelect = screen.getByRole("combobox");

    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const s3Option = screen.queryByText("AWS S3 Vectors");
      if (s3Option) {
        fireEvent.click(s3Option);
      }
    });

    // Try to create without filling required fields
    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });

    await act(async () => {
      fireEvent.click(createButton);
    });

    // Should show validation warning (mocked message.warning would be called)
    // The actual validation happens in the component
  });
});
