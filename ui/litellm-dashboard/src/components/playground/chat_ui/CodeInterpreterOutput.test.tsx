import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import CodeInterpreterOutput from "./CodeInterpreterOutput";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

global.fetch = vi.fn();

describe("CodeInterpreterOutput", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    URL.createObjectURL = vi.fn((blob) => `blob:${blob}`);
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should render", () => {
    render(<CodeInterpreterOutput code="print('hello')" accessToken="test-token" />);

    expect(screen.getByText("Python Code Executed")).toBeInTheDocument();
  });

  it("should display code in syntax highlighter", async () => {
    const user = userEvent.setup();
    const code = "print('hello world')";
    const { container } = render(<CodeInterpreterOutput code={code} accessToken="test-token" />);

    expect(screen.getByText("Python Code Executed")).toBeInTheDocument();

    const collapseHeader = screen.getByRole("button");
    await user.click(collapseHeader);

    await waitFor(() => {
      const codeElement = container.querySelector("code.language-python");
      expect(codeElement).toBeInTheDocument();
      expect(codeElement?.textContent).toContain(code);
    });
  });

  it("should fetch and display images from annotations", async () => {
    const mockBlob = new Blob(["image data"], { type: "image/png" });
    const mockResponse = {
      ok: true,
      blob: vi.fn().mockResolvedValue(mockBlob),
    };

    (global.fetch as any).mockResolvedValue(mockResponse);

    const annotations = [
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-1",
        filename: "chart.png",
        start_index: 0,
        end_index: 10,
      },
    ];

    render(
      <CodeInterpreterOutput
        code="import matplotlib.pyplot as plt"
        annotations={annotations}
        accessToken="test-token"
      />,
    );

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "https://example.com/v1/containers/container-1/files/file-1/content",
        expect.objectContaining({
          headers: {
            Authorization: "Bearer test-token",
          },
        }),
      );
    });

    await waitFor(() => {
      expect(screen.getByText("chart.png")).toBeInTheDocument();
    });
  });

  it("should show loading state while fetching images", async () => {
    const mockBlob = new Blob(["image data"], { type: "image/png" });
    let resolveBlob: (value: Blob) => void;
    const blobPromise = new Promise<Blob>((resolve) => {
      resolveBlob = resolve;
    });

    const mockResponse = {
      ok: true,
      blob: vi.fn().mockReturnValue(blobPromise),
    };

    (global.fetch as any).mockResolvedValue(mockResponse);

    const annotations = [
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-1",
        filename: "chart.png",
        start_index: 0,
        end_index: 10,
      },
    ];

    render(
      <CodeInterpreterOutput
        code="import matplotlib.pyplot as plt"
        annotations={annotations}
        accessToken="test-token"
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Loading image...")).toBeInTheDocument();
    });

    resolveBlob!(mockBlob);
    await waitFor(() => {
      expect(screen.queryByText("Loading image...")).not.toBeInTheDocument();
    });
  });

  it("should handle download for image files", async () => {
    const user = userEvent.setup();
    const mockBlob = new Blob(["image data"], { type: "image/png" });
    const mockResponse = {
      ok: true,
      blob: vi.fn().mockResolvedValue(mockBlob),
    };

    (global.fetch as any).mockResolvedValue(mockResponse);

    const annotations = [
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-1",
        filename: "chart.png",
        start_index: 0,
        end_index: 10,
      },
    ];

    const createElementSpy = vi.spyOn(document, "createElement");
    const appendChildSpy = vi.spyOn(document.body, "appendChild");
    const removeChildSpy = vi.spyOn(document.body, "removeChild");

    render(
      <CodeInterpreterOutput
        code="import matplotlib.pyplot as plt"
        annotations={annotations}
        accessToken="test-token"
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("chart.png")).toBeInTheDocument();
    });

    const downloadButton = screen.getByText("Download");
    await user.click(downloadButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "https://example.com/v1/containers/container-1/files/file-1/content",
        expect.objectContaining({
          headers: {
            Authorization: "Bearer test-token",
          },
        }),
      );
    });

    createElementSpy.mockRestore();
    appendChildSpy.mockRestore();
    removeChildSpy.mockRestore();
  });

  it("should handle download for non-image files", async () => {
    const user = userEvent.setup();
    const mockBlob = new Blob(["file data"], { type: "text/plain" });
    const mockResponse = {
      ok: true,
      blob: vi.fn().mockResolvedValue(mockBlob),
    };

    (global.fetch as any).mockResolvedValue(mockResponse);

    const annotations = [
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-1",
        filename: "data.csv",
        start_index: 0,
        end_index: 10,
      },
    ];

    render(<CodeInterpreterOutput code="import pandas as pd" annotations={annotations} accessToken="test-token" />);

    await waitFor(() => {
      expect(screen.getByText("data.csv")).toBeInTheDocument();
    });

    const downloadButton = screen.getByText("data.csv").closest("button");
    expect(downloadButton).toBeInTheDocument();
    if (downloadButton) {
      await user.click(downloadButton);
    }

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "https://example.com/v1/containers/container-1/files/file-1/content",
        expect.objectContaining({
          headers: {
            Authorization: "Bearer test-token",
          },
        }),
      );
    });
  });

  it("should return null when no code and no annotations", () => {
    const { container } = render(<CodeInterpreterOutput accessToken="test-token" />);

    expect(container.firstChild).toBeNull();
  });

  it("should handle multiple image formats", async () => {
    const mockBlob = new Blob(["image data"], { type: "image/png" });
    const mockResponse = {
      ok: true,
      blob: vi.fn().mockResolvedValue(mockBlob),
    };

    (global.fetch as any).mockResolvedValue(mockResponse);

    const annotations = [
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-1",
        filename: "image.png",
        start_index: 0,
        end_index: 10,
      },
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-2",
        filename: "image.jpg",
        start_index: 0,
        end_index: 10,
      },
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-3",
        filename: "image.jpeg",
        start_index: 0,
        end_index: 10,
      },
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-4",
        filename: "image.gif",
        start_index: 0,
        end_index: 10,
      },
    ];

    render(
      <CodeInterpreterOutput
        code="import matplotlib.pyplot as plt"
        annotations={annotations}
        accessToken="test-token"
      />,
    );

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(4);
    });
  });

  it("should handle fetch errors gracefully", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    (global.fetch as any).mockRejectedValue(new Error("Network error"));

    const annotations = [
      {
        type: "container_file_citation" as const,
        container_id: "container-1",
        file_id: "file-1",
        filename: "chart.png",
        start_index: 0,
        end_index: 10,
      },
    ];

    render(
      <CodeInterpreterOutput
        code="import matplotlib.pyplot as plt"
        annotations={annotations}
        accessToken="test-token"
      />,
    );

    await waitFor(() => {
      expect(consoleErrorSpy).toHaveBeenCalled();
    });

    consoleErrorSpy.mockRestore();
  });
});
