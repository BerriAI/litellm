import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { BlogDropdown } from "../BlogDropdown";
import { useDisableShowBlog } from "@/app/(dashboard)/hooks/useDisableShowBlog";

// Mock hooks
vi.mock("@/app/(dashboard)/hooks/useDisableShowBlog", () => ({
  useDisableShowBlog: vi.fn(() => false),
}));

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://localhost:4000",
}));

const SAMPLE_POSTS = {
  posts: [
    {
      title: "Test Post 1",
      description: "First test post description.",
      date: "2026-02-01",
      url: "https://www.litellm.ai/blog/test-1",
    },
    {
      title: "Test Post 2",
      description: "Second test post description.",
      date: "2026-01-15",
      url: "https://www.litellm.ai/blog/test-2",
    },
  ],
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("BlogDropdown", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the Blog button", async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => SAMPLE_POSTS,
    });

    render(<BlogDropdown />, { wrapper: createWrapper() });
    expect(screen.getByText("Blog")).toBeInTheDocument();
  });

  it("shows posts on success", async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => SAMPLE_POSTS,
    });

    render(<BlogDropdown />, { wrapper: createWrapper() });

    // Open the dropdown
    fireEvent.click(screen.getByText("Blog"));

    await waitFor(() => {
      expect(screen.getByText("Test Post 1")).toBeInTheDocument();
      expect(screen.getByText("Test Post 2")).toBeInTheDocument();
    });
  });

  it("shows error message and Retry button on fetch failure", async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new Error("Network error"));

    render(<BlogDropdown />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText("Blog"));

    await waitFor(() => {
      expect(screen.getByText(/Failed to load blog posts/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
    });
  });

  it("calls refetch when Retry is clicked", async () => {
    global.fetch = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network error"))
      .mockResolvedValueOnce({ ok: true, json: async () => SAMPLE_POSTS });

    render(<BlogDropdown />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText("Blog"));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => {
      expect(screen.getByText("Test Post 1")).toBeInTheDocument();
    });
  });

  it("returns null when useDisableShowBlog is true", () => {
    vi.mocked(useDisableShowBlog).mockReturnValue(true);

    const { container } = render(<BlogDropdown />, { wrapper: createWrapper() });
    expect(container.firstChild).toBeNull();
  });
});
