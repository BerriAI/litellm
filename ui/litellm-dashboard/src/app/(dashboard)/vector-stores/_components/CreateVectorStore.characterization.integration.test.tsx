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
  const option = await screen.findByRole("option", { name: label });
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

  it("submits OpenAI ingestion using a stored credential name without raw credentials", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickProvider("OpenAI");

    fireEvent.change(screen.getByRole("textbox", { name: "Stored Credential Name" }), {
      target: { value: "openai-ingestion" },
    });
    expect(screen.queryByLabelText("API Key")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("API Base")).not.toBeInTheDocument();
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    expect(networking.ragIngestCall).toHaveBeenCalledWith(
      "test-token",
      expect.any(File),
      "openai",
      undefined,
      undefined,
      undefined,
      { litellm_credential_name: "openai-ingestion" },
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
    await pickProvider("Amazon S3 Vectors");

    const modelInput = screen.getAllByRole("combobox").at(-1) as HTMLElement;
    await user.click(modelInput);
    expect((await screen.findAllByText("text-embedding-3-small")).at(-1)).toBeInTheDocument();
    expect(screen.queryByText("gpt-5")).not.toBeInTheDocument();

    fireEvent.change(modelInput, { target: { value: "large" } });
    await user.click((await screen.findAllByText("text-embedding-3-large")).at(-1) as HTMLElement);
    fireEvent.change(screen.getByLabelText(/Vector Bucket Name/), { target: { value: "test-bucket" } });
    fireEvent.change(screen.getByLabelText(/AWS Region/), { target: { value: "us-west-2" } });
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(networking.ragIngestCall).mock.calls.at(-1)?.[6]).toEqual({
      embedding_model: "text-embedding-3-large",
      vector_bucket_name: "test-bucket",
      aws_region_name: "us-west-2",
    });
  });

  it("blocks the submit when a required provider field is missing", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickProvider("Amazon S3 Vectors");
    await clickCreate();

    expect(networking.ragIngestCall).not.toHaveBeenCalled();
  });

  it("clears provider settings when switching providers", async () => {
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickProvider("Amazon S3 Vectors");
    fireEvent.change(screen.getByRole("textbox", { name: "Stored Credential Name" }), {
      target: { value: "aws-ingestion" },
    });
    fireEvent.change(screen.getByLabelText(/Vector Bucket Name/), { target: { value: "test-bucket" } });
    await pickProvider("OpenAI");
    await clickCreate();

    await waitFor(() =>
      expect(networking.ragIngestCall).toHaveBeenCalledWith(
        "test-token",
        expect.any(File),
        "openai",
        undefined,
        undefined,
        undefined,
        {},
      ),
    );
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
