import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import CreateVectorStore from "./CreateVectorStore";
import * as networking from "@/components/networking";
import * as fetchModels from "@/components/llm_calls/fetch_models";

vi.mock("@/components/networking", () => ({
  ragIngestCall: vi.fn(),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

vi.mock("@/components/vector_store_providers", () => ({
  VectorStoreProviders: {
    BEDROCK: "Amazon Bedrock",
    S3Vectors: "AWS S3 Vectors",
    PGVECTOR: "PG Vector",
  },
  vectorStoreProviderMap: {
    BEDROCK: "bedrock",
    S3Vectors: "s3_vectors",
    PGVECTOR: "pg_vector",
  },
  vectorStoreProviderLogoMap: {
    "Amazon Bedrock": "https://example.com/bedrock.png",
    "AWS S3 Vectors": "https://example.com/aws.png",
    "PG Vector": "https://example.com/pg.png",
  },
  getProviderSpecificFields: vi.fn((provider: string) => {
    if (provider === "pg_vector") {
      return [
        {
          name: "api_base",
          label: "API Base",
          tooltip: "Base URL of the pgvector server",
          placeholder: "http://localhost:8000",
          required: true,
          type: "text",
        },
        {
          name: "api_key",
          label: "API Key",
          tooltip: "Secret for the pgvector server",
          placeholder: "sk-...",
          required: false,
          type: "password",
        },
        {
          name: "embedding_model",
          label: "Embedding Model",
          tooltip: "Model used to embed documents",
          placeholder: "text-embedding-3-small",
          required: false,
          type: "select",
        },
      ];
    }
    return [];
  }),
}));

const uploadFile = async (name = "test.pdf") => {
  const file = new File(["test content"], name, { type: "application/pdf" });
  const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;
  await act(async () => {
    fireEvent.change(uploadInput, { target: { files: [file] } });
  });
  await screen.findByText(/Uploaded Documents \(1\)/);
};

const pickProvider = async (label: string) => {
  const user = userEvent.setup();
  const trigger = screen.getAllByRole("combobox")[0];
  await user.click(trigger);
  const option = await screen.findByText(label);
  await user.click(option);
};

const clickCreate = async () => {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: /Create Vector Store/i }));
  });
};

describe("CreateVectorStore submit payload characterization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchModels.fetchAvailableModels).mockResolvedValue([]);
    vi.mocked(networking.ragIngestCall).mockResolvedValue({
      id: "test-id",
      status: "completed",
      vector_store_id: "vs_123",
      file_id: "file_123",
    });
  });

  it("sends undefined, not empty string, for an untouched name and description", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    expect(networking.ragIngestCall).toHaveBeenCalledWith(
      "test-token",
      expect.any(File),
      "bedrock",
      undefined,
      undefined,
      undefined,
      {},
    );
  });

  it("forwards the typed name and description verbatim", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();

    fireEvent.change(screen.getByPlaceholderText("e.g., Product Documentation, Customer Support KB"), {
      target: { value: "  Product Docs  " },
    });
    fireEvent.change(screen.getByPlaceholderText("e.g., Contains all product documentation and user guides"), {
      target: { value: "All the guides" },
    });
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    expect(networking.ragIngestCall).toHaveBeenCalledWith(
      "test-token",
      expect.any(File),
      "bedrock",
      undefined,
      "  Product Docs  ",
      "All the guides",
      {},
    );
  });

  it("accumulates provider-specific fields into the providerParams argument", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickProvider("PG Vector");

    fireEvent.change(screen.getByPlaceholderText("http://localhost:8000"), {
      target: { value: "http://pg.internal:8000" },
    });
    fireEvent.change(screen.getByPlaceholderText("sk-..."), { target: { value: "sk-secret" } });
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    expect(networking.ragIngestCall).toHaveBeenCalledWith(
      "test-token",
      expect.any(File),
      "pg_vector",
      undefined,
      undefined,
      undefined,
      { api_base: "http://pg.internal:8000", api_key: "sk-secret" },
    );
  });

  it("loads the S3 embedding models, filters them by typing and sends the chosen one", async () => {
    vi.mocked(fetchModels.fetchAvailableModels).mockResolvedValue([
      { model_group: "text-embedding-3-small", mode: "embedding" },
      { model_group: "text-embedding-3-large", mode: "embedding" },
      { model_group: "gpt-5", mode: "chat" },
    ] as Awaited<ReturnType<typeof fetchModels.fetchAvailableModels>>);
    const user = userEvent.setup();
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickProvider("AWS S3 Vectors");

    const modelInput = screen.getAllByRole("combobox").at(-1) as HTMLElement;
    await user.click(modelInput);
    expect((await screen.findAllByText("text-embedding-3-small")).at(-1)).toBeInTheDocument();
    expect(screen.queryByText("gpt-5")).not.toBeInTheDocument();

    fireEvent.change(modelInput, { target: { value: "large" } });
    await user.click((await screen.findAllByText("text-embedding-3-large")).at(-1) as HTMLElement);
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(networking.ragIngestCall).mock.calls.at(-1)?.[6]).toEqual({
      embedding_model: "text-embedding-3-large",
    });
  });

  it("blocks the submit when a required provider field is missing", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickProvider("PG Vector");
    await clickCreate();

    expect(networking.ragIngestCall).not.toHaveBeenCalled();
  });

  it("renders the password provider field as a masked input", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await pickProvider("PG Vector");

    expect(screen.getByPlaceholderText("sk-...")).toHaveAttribute("type", "password");
  });

  it("reuses the vector store id returned by the first ingest for later documents", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    const files = [
      new File(["a"], "a.pdf", { type: "application/pdf" }),
      new File(["b"], "b.pdf", { type: "application/pdf" }),
    ];
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await act(async () => {
      fireEvent.change(uploadInput, { target: { files } });
    });
    await screen.findByText(/Uploaded Documents \(2\)/);
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(2));
    expect(vi.mocked(networking.ragIngestCall).mock.calls[0][3]).toBeUndefined();
    expect(vi.mocked(networking.ragIngestCall).mock.calls[1][3]).toBe("vs_123");
  });

  it("does not submit at all when no document has been uploaded", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await clickCreate();

    expect(networking.ragIngestCall).not.toHaveBeenCalled();
  });
});
