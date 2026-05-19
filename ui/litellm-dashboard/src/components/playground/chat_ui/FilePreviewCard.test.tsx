import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import FilePreviewCard from "./FilePreviewCard";

function makeFile(name: string): File {
  return new File(["dummy"], name, { type: "application/octet-stream" });
}

describe("FilePreviewCard", () => {
  it("should render", () => {
    render(
      <FilePreviewCard file={makeFile("photo.png")} previewUrl={null} onRemove={vi.fn()} />
    );
    expect(screen.getByText("photo.png")).toBeInTheDocument();
  });

  it("should display the file name", () => {
    render(
      <FilePreviewCard file={makeFile("my-screenshot.jpg")} previewUrl={null} onRemove={vi.fn()} />
    );
    expect(screen.getByText("my-screenshot.jpg")).toBeInTheDocument();
  });

  it("should show 'Image' label for non-PDF files", () => {
    render(
      <FilePreviewCard file={makeFile("photo.png")} previewUrl="blob:http://localhost/abc" onRemove={vi.fn()} />
    );
    expect(screen.getByText("Image")).toBeInTheDocument();
  });

  it("should show 'PDF' label for PDF files", () => {
    render(
      <FilePreviewCard file={makeFile("report.pdf")} previewUrl={null} onRemove={vi.fn()} />
    );
    expect(screen.getByText("PDF")).toBeInTheDocument();
  });

  it("should render an image preview when the file is not a PDF", () => {
    render(
      <FilePreviewCard file={makeFile("photo.png")} previewUrl="blob:http://localhost/abc" onRemove={vi.fn()} />
    );
    expect(screen.getByAltText("Upload preview")).toBeInTheDocument();
  });

  it("should not render an image preview when the file is a PDF", () => {
    render(
      <FilePreviewCard file={makeFile("doc.PDF")} previewUrl={null} onRemove={vi.fn()} />
    );
    expect(screen.queryByAltText("Upload preview")).not.toBeInTheDocument();
  });

  it("should call onRemove when the remove button is clicked", async () => {
    const onRemove = vi.fn();
    const user = userEvent.setup();
    render(
      <FilePreviewCard file={makeFile("photo.png")} previewUrl={null} onRemove={onRemove} />
    );
    await user.click(screen.getByRole("button"));
    expect(onRemove).toHaveBeenCalledOnce();
  });

  it("should treat .PDF (uppercase) as a PDF file", () => {
    render(
      <FilePreviewCard file={makeFile("REPORT.PDF")} previewUrl={null} onRemove={vi.fn()} />
    );
    expect(screen.getByText("PDF")).toBeInTheDocument();
    expect(screen.queryByAltText("Upload preview")).not.toBeInTheDocument();
  });
});
