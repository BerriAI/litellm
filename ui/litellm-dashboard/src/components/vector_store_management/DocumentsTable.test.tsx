import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import DocumentsTable from "./DocumentsTable";
import { DocumentUpload } from "./types";

// Mock antd message
vi.mock("antd", async () => {
  const actual = await vi.importActual("antd");
  return {
    ...actual,
    message: {
      success: vi.fn(),
    },
  };
});

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

  it("should render the table successfully", () => {
    const onRemove = vi.fn();
    render(<DocumentsTable documents={mockDocuments} onRemove={onRemove} />);

    expect(screen.getByText("test1.pdf")).toBeInTheDocument();
    expect(screen.getByText("test2.txt")).toBeInTheDocument();
    expect(screen.getByText("test3.docx")).toBeInTheDocument();
  });

  it("should display correct status badges", () => {
    const onRemove = vi.fn();
    render(<DocumentsTable documents={mockDocuments} onRemove={onRemove} />);

    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("Uploading")).toBeInTheDocument();
    expect(screen.getByText("Error")).toBeInTheDocument();
  });

  it("should display file sizes", () => {
    const onRemove = vi.fn();
    render(<DocumentsTable documents={mockDocuments} onRemove={onRemove} />);

    expect(screen.getByText(/1000.00 KB/)).toBeInTheDocument();
    expect(screen.getByText(/1.95 MB/)).toBeInTheDocument();
    expect(screen.getByText(/500.00 KB/)).toBeInTheDocument();
  });

  it("should call onRemove when delete button is clicked", () => {
    const onRemove = vi.fn();
    render(<DocumentsTable documents={mockDocuments} onRemove={onRemove} />);

    const deleteButtons = screen.getAllByLabelText(/delete/i);

    act(() => {
      fireEvent.click(deleteButtons[0]);
    });

    expect(onRemove).toHaveBeenCalledWith("1");
  });

  it("should show empty state when no documents", () => {
    const onRemove = vi.fn();
    render(<DocumentsTable documents={[]} onRemove={onRemove} />);

    expect(screen.getByText(/No documents uploaded yet/)).toBeInTheDocument();
  });

  it("should have action buttons for each document", () => {
    const onRemove = vi.fn();
    render(<DocumentsTable documents={mockDocuments} onRemove={onRemove} />);

    // Each document should have 3 action buttons (view, copy, delete)
    const viewButtons = screen.getAllByLabelText(/eye/i);
    const copyButtons = screen.getAllByLabelText(/copy/i);
    const deleteButtons = screen.getAllByLabelText(/delete/i);

    expect(viewButtons).toHaveLength(3);
    expect(copyButtons).toHaveLength(3);
    expect(deleteButtons).toHaveLength(3);
  });
});
