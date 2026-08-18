import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DocumentUpload } from "@/components/vector_store_management/types";

import DocumentsTable from "./DocumentsTable";

describe("DocumentsTable", () => {
  const mockDocuments: DocumentUpload[] = [
    {
      uid: "1",
      name: "test1.pdf",
      status: "done",
      size: 1024000,
      type: "application/pdf",
    },
    {
      uid: "2",
      name: "test2.txt",
      status: "uploading",
      size: 2048000,
      type: "text/plain",
    },
    {
      uid: "3",
      name: "test3.docx",
      status: "error",
      size: 512000,
      type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every document row", () => {
    render(<DocumentsTable documents={mockDocuments} onRemove={vi.fn()} />);

    expect(screen.getByText("test1.pdf")).toBeInTheDocument();
    expect(screen.getByText("test2.txt")).toBeInTheDocument();
    expect(screen.getByText("test3.docx")).toBeInTheDocument();
  });

  it("should display correct status badges", () => {
    render(<DocumentsTable documents={mockDocuments} onRemove={vi.fn()} />);

    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("Uploading")).toBeInTheDocument();
    expect(screen.getByText("Error")).toBeInTheDocument();
  });

  it("should display file sizes", () => {
    render(<DocumentsTable documents={mockDocuments} onRemove={vi.fn()} />);

    expect(screen.getByText(/1000.00 KB/)).toBeInTheDocument();
    expect(screen.getByText(/1.95 MB/)).toBeInTheDocument();
    expect(screen.getByText(/500.00 KB/)).toBeInTheDocument();
  });

  it("should call onRemove through the actions menu", async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    render(<DocumentsTable documents={mockDocuments} onRemove={onRemove} />);

    await user.click(screen.getByTestId("document-actions-1"));
    await user.click(await screen.findByTestId("document-action-remove"));

    expect(onRemove).toHaveBeenCalledWith("1");
  });

  it("should copy the document ID through the actions menu", async () => {
    const user = userEvent.setup();
    render(<DocumentsTable documents={mockDocuments} onRemove={vi.fn()} />);

    await user.click(screen.getByTestId("document-actions-2"));
    await user.click(await screen.findByTestId("document-action-copy"));

    expect(await window.navigator.clipboard.readText()).toBe("2");
  });

  it("should show the empty state when no documents", () => {
    render(<DocumentsTable documents={[]} onRemove={vi.fn()} />);

    expect(screen.getByText("No documents uploaded yet")).toBeInTheDocument();
    expect(screen.getByText("Upload documents above to get started.")).toBeInTheDocument();
  });

  it("should render one actions menu per document", () => {
    render(<DocumentsTable documents={mockDocuments} onRemove={vi.fn()} />);

    for (const doc of mockDocuments) {
      expect(screen.getByTestId(`document-actions-${doc.uid}`)).toBeInTheDocument();
    }
  });
});
