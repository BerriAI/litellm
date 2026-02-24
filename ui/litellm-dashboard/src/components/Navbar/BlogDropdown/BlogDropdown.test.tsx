import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../../tests/test-utils";
import { BlogDropdown } from "./BlogDropdown";

let mockDisableBlogPosts = false;
let mockRefetch = vi.fn();
let mockUseBlogPostsResult: {
  data: { posts: { title: string; date: string; description: string; url: string }[] } | null | undefined;
  isLoading: boolean;
  isError: boolean;
  refetch: () => void;
} = {
  data: undefined,
  isLoading: false,
  isError: false,
  refetch: mockRefetch,
};

vi.mock("@/app/(dashboard)/hooks/useDisableBlogPosts", () => ({
  useDisableBlogPosts: () => mockDisableBlogPosts,
}));

vi.mock("@/app/(dashboard)/hooks/blogPosts/useBlogPosts", () => ({
  useBlogPosts: () => mockUseBlogPostsResult,
}));

const MOCK_POSTS = [
  { title: "Post One", date: "2026-02-01", description: "Description one", url: "https://example.com/1" },
  { title: "Post Two", date: "2026-02-02", description: "Description two", url: "https://example.com/2" },
  { title: "Post Three", date: "2026-02-03", description: "Description three", url: "https://example.com/3" },
  { title: "Post Four", date: "2026-02-04", description: "Description four", url: "https://example.com/4" },
  { title: "Post Five", date: "2026-02-05", description: "Description five", url: "https://example.com/5" },
  { title: "Post Six", date: "2026-02-06", description: "Description six", url: "https://example.com/6" },
];

async function openDropdown() {
  const user = userEvent.setup();
  await user.hover(screen.getByRole("button", { name: /blog/i }));
}

describe("BlogDropdown", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDisableBlogPosts = false;
    mockRefetch = vi.fn();
    mockUseBlogPostsResult = {
      data: undefined,
      isLoading: false,
      isError: false,
      refetch: mockRefetch,
    };
  });

  describe("when blog posts are disabled", () => {
    it("should render nothing", () => {
      mockDisableBlogPosts = true;
      const { container } = renderWithProviders(<BlogDropdown />);
      expect(container).toBeEmptyDOMElement();
    });
  });

  describe("when blog posts are enabled", () => {
    it("should render the Blog trigger button", () => {
      renderWithProviders(<BlogDropdown />);
      expect(screen.getByRole("button", { name: /blog/i })).toBeInTheDocument();
    });

    describe("loading state", () => {
      it("should show a loading spinner", async () => {
        mockUseBlogPostsResult = { ...mockUseBlogPostsResult, isLoading: true };
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(document.querySelector(".anticon-loading")).toBeInTheDocument();
        });
      });
    });

    describe("error state", () => {
      beforeEach(() => {
        mockUseBlogPostsResult = { ...mockUseBlogPostsResult, isError: true };
      });

      it("should show an error message", async () => {
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByText("Failed to load posts")).toBeInTheDocument();
        });
      });

      it("should show a Retry button", async () => {
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
        });
      });

      it("should call refetch when Retry is clicked", async () => {
        const user = userEvent.setup();
        renderWithProviders(<BlogDropdown />);

        await user.hover(screen.getByRole("button", { name: /blog/i }));

        await waitFor(() => {
          expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
        });

        await user.click(screen.getByRole("button", { name: /retry/i }));

        expect(mockRefetch).toHaveBeenCalledTimes(1);
      });
    });

    describe("empty state", () => {
      it("should show 'No posts available' when data is null", async () => {
        mockUseBlogPostsResult = { ...mockUseBlogPostsResult, data: null };
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByText("No posts available")).toBeInTheDocument();
        });
      });

      it("should show 'No posts available' when posts array is empty", async () => {
        mockUseBlogPostsResult = { ...mockUseBlogPostsResult, data: { posts: [] } };
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByText("No posts available")).toBeInTheDocument();
        });
      });
    });

    describe("with posts", () => {
      beforeEach(() => {
        mockUseBlogPostsResult = { ...mockUseBlogPostsResult, data: { posts: MOCK_POSTS.slice(0, 3) } };
      });

      it("should render post titles", async () => {
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByText("Post One")).toBeInTheDocument();
          expect(screen.getByText("Post Two")).toBeInTheDocument();
          expect(screen.getByText("Post Three")).toBeInTheDocument();
        });
      });

      it("should render post descriptions", async () => {
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByText("Description one")).toBeInTheDocument();
        });
      });

      it("should render post links with correct attributes", async () => {
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          const link = screen.getByRole("link", { name: /post one/i });
          expect(link).toHaveAttribute("href", "https://example.com/1");
          expect(link).toHaveAttribute("target", "_blank");
          expect(link).toHaveAttribute("rel", "noopener noreferrer");
        });
      });

      it("should render formatted post dates", async () => {
        mockUseBlogPostsResult = {
          ...mockUseBlogPostsResult,
          data: { posts: [{ title: "Date Post", date: "2026-02-15", description: "Desc", url: "https://example.com" }] },
        };
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByText("Feb 15, 2026")).toBeInTheDocument();
        });
      });

      it("should render the 'View all posts' link", async () => {
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          const viewAllLink = screen.getByRole("link", { name: /view all posts/i });
          expect(viewAllLink).toHaveAttribute("href", "https://docs.litellm.ai/blog");
          expect(viewAllLink).toHaveAttribute("target", "_blank");
          expect(viewAllLink).toHaveAttribute("rel", "noopener noreferrer");
        });
      });
    });

    describe("post limit", () => {
      it("should render at most 5 posts when more than 5 are provided", async () => {
        mockUseBlogPostsResult = { ...mockUseBlogPostsResult, data: { posts: MOCK_POSTS } };
        renderWithProviders(<BlogDropdown />);

        await openDropdown();

        await waitFor(() => {
          expect(screen.getByText("Post One")).toBeInTheDocument();
          expect(screen.getByText("Post Five")).toBeInTheDocument();
          expect(screen.queryByText("Post Six")).not.toBeInTheDocument();
        });
      });
    });
  });
});
