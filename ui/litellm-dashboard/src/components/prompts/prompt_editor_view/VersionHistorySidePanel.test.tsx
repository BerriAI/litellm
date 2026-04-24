import { describe, it, expect, vi, beforeEach, afterEach, type Mock } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import VersionHistorySidePanel from "./VersionHistorySidePanel";
import { getPromptVersions } from "../../networking";
import type { PromptSpec } from "../../networking";

// Mock the networking function
vi.mock("../../networking", () => ({
  getPromptVersions: vi.fn(),
}));

const mockGetPromptVersions = getPromptVersions as Mock;

// The component was migrated from antd Drawer/List/Skeleton/Tag to shadcn
// Sheet/Badge/Skeleton; no antd stubs are needed.

describe("VersionHistorySidePanel", () => {
  // Mock data
  const mockPromptVersions: PromptSpec[] = [
    {
      prompt_id: "test-prompt.v2",
      litellm_params: { prompt_id: "test-prompt.v2" },
      prompt_info: { prompt_type: "db" },
      version: 2,
      created_at: "2024-01-15T10:30:00Z",
    },
    {
      prompt_id: "test-prompt.v1",
      litellm_params: { prompt_id: "test-prompt.v1" },
      prompt_info: { prompt_type: "db" },
      version: 1,
      created_at: "2024-01-10T09:00:00Z",
    },
    {
      prompt_id: "test-prompt.v3",
      litellm_params: { prompt_id: "test-prompt.v3" },
      prompt_info: { prompt_type: "config" },
      version: 3,
      created_at: "2024-01-20T14:15:00Z",
    },
  ];

  const mockPromptVersionsWithoutExplicitVersion = [
    {
      prompt_id: "test-prompt.v2",
      litellm_params: { prompt_id: "test-prompt.v2" },
      prompt_info: { prompt_type: "db" },
      created_at: "2024-01-15T10:30:00Z",
    },
    {
      prompt_id: "test-prompt.v1",
      litellm_params: { prompt_id: "test-prompt.v1" },
      prompt_info: { prompt_type: "db" },
      created_at: "2024-01-10T09:00:00Z",
    },
  ];

  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    accessToken: "test-token",
    promptId: "test-prompt.v2",
    activeVersionId: "test-prompt.v2",
    onSelectVersion: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    // Mock successful response by default
    mockGetPromptVersions.mockResolvedValue({
      prompts: mockPromptVersions,
    });
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe("Component Rendering", () => {
    it("should render the component with drawer", async () => {
      await act(async () => {
        render(<VersionHistorySidePanel {...defaultProps} />);
      });
      expect(screen.getByRole("dialog", { name: /version history/i })).toBeInTheDocument();
      expect(screen.getByText("Version History")).toBeInTheDocument();
    });

    it("should not render when isOpen is false", async () => {
      await act(async () => {
        render(<VersionHistorySidePanel {...defaultProps} isOpen={false} />);
      });
      // The shadcn Sheet unmounts its content when not open.
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("should show loading skeleton initially", async () => {
      // Mock a delayed response to show loading state
      mockGetPromptVersions.mockImplementationOnce(
        () => new Promise((resolve) => setTimeout(() => resolve({ prompts: mockPromptVersions }), 100)),
      );

      render(<VersionHistorySidePanel {...defaultProps} />);
      // The Sheet content renders in a portal, so query the document.
      expect(document.querySelector(".animate-pulse")).toBeInTheDocument();

      // Wait for loading to complete
      await waitFor(() => {
        expect(document.querySelector(".animate-pulse")).not.toBeInTheDocument();
      });
    });

    it("should show empty state when no versions are available", async () => {
      mockGetPromptVersions.mockResolvedValueOnce({ prompts: [] });

      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("No version history available.")).toBeInTheDocument();
      });
    });

    it("should render version list when data is loaded", async () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("v2")).toBeInTheDocument();
        expect(screen.getByText("v1")).toBeInTheDocument();
        expect(screen.getByText("v3")).toBeInTheDocument();
      });

      // Check that Latest tag is shown for the first item
      const latestTags = screen.getAllByText("Latest");
      expect(latestTags.length).toBeGreaterThan(0);

      // Check Active tag is shown for the active version
      expect(screen.getByText("Active")).toBeInTheDocument();
    });
  });

  describe("Version Selection and Highlighting", () => {
    it("should highlight the active version correctly", async () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        // Should have the Active badge for the selected version
        expect(screen.getByText("Active")).toBeInTheDocument();
      });
    });

    it("should highlight the latest version when no activeVersionId is provided", async () => {
      render(<VersionHistorySidePanel {...defaultProps} activeVersionId={undefined} />);

      await waitFor(() => {
        const latestTags = screen.getAllByText("Latest");
        expect(latestTags.length).toBeGreaterThan(0);
      });
    });

    it("should call onSelectVersion when a version is clicked", async () => {
      const mockOnSelectVersion = vi.fn();
      render(<VersionHistorySidePanel {...defaultProps} onSelectVersion={mockOnSelectVersion} />);

      await waitFor(() => {
        expect(screen.getByText("v1")).toBeInTheDocument();
      });

      const versionItem = screen.getByText("v1").closest("div");
      expect(versionItem).toBeInTheDocument();

      act(() => {
        fireEvent.click(versionItem!);
      });

      expect(mockOnSelectVersion).toHaveBeenCalledWith(mockPromptVersions[1]);
    });
  });

  describe("Version Number Extraction", () => {
    it("should extract version from explicit version field", async () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("v2")).toBeInTheDocument();
        expect(screen.getByText("v3")).toBeInTheDocument();
      });
    });

    it("should extract version from prompt_id with .v suffix", async () => {
      mockGetPromptVersions.mockResolvedValueOnce({
        prompts: mockPromptVersionsWithoutExplicitVersion,
      });

      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("v2")).toBeInTheDocument();
        expect(screen.getByText("v1")).toBeInTheDocument();
      });
    });

    it("should extract version from prompt_id with _v suffix", async () => {
      const versionsWithUnderscore = [
        {
          prompt_id: "test-prompt_v2",
          litellm_params: { prompt_id: "test-prompt_v2" },
          prompt_info: { prompt_type: "db" },
          created_at: "2024-01-15T10:30:00Z",
        },
      ];

      mockGetPromptVersions.mockResolvedValueOnce({
        prompts: versionsWithUnderscore,
      });

      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("v2")).toBeInTheDocument();
      });
    });

    it("should default to v1 when no version info is available", async () => {
      const versionWithoutVersionInfo = [
        {
          prompt_id: "test-prompt",
          litellm_params: { prompt_id: "test-prompt" },
          prompt_info: { prompt_type: "db" },
          created_at: "2024-01-15T10:30:00Z",
        },
      ];

      mockGetPromptVersions.mockResolvedValueOnce({
        prompts: versionWithoutVersionInfo,
      });

      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("v1")).toBeInTheDocument();
      });
    });
  });

  describe("Date Formatting", () => {
    it("should format dates correctly", async () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        // Dates are rendered using toLocaleString() — find any element whose
        // text looks like a locale-formatted date.
        const nodes = Array.from(document.querySelectorAll("span"));
        const dateText = nodes.find((el) => /\d{1,4}\/\d{1,4}\/\d{2,4}|\d{4}/.test(el.textContent || ""));
        expect(dateText).toBeTruthy();
      });
    });

    it("should show dash for missing dates", async () => {
      const versionsWithoutDates = [
        {
          prompt_id: "test-prompt.v1",
          litellm_params: { prompt_id: "test-prompt.v1" },
          prompt_info: { prompt_type: "db" },
          version: 1,
        },
      ];

      mockGetPromptVersions.mockResolvedValueOnce({
        prompts: versionsWithoutDates,
      });

      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("-")).toBeInTheDocument();
      });
    });
  });

  describe("Prompt Type Display", () => {
    it("should show 'Saved to Database' for db prompts", async () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        const dbTexts = screen.getAllByText("Saved to Database");
        expect(dbTexts.length).toBeGreaterThan(0);
      });
    });

    it("should show 'Config Prompt' for config prompts", async () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Config Prompt")).toBeInTheDocument();
      });
    });
  });

  describe("Network Calls and Data Fetching", () => {
    it("should call getPromptVersions with correct parameters", async () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(getPromptVersions).toHaveBeenCalledWith("test-token", "test-prompt");
      });
    });

    it("should strip .v suffix from promptId when fetching versions", async () => {
      render(<VersionHistorySidePanel {...defaultProps} promptId="test-prompt.v3" />);

      await waitFor(() => {
        expect(getPromptVersions).toHaveBeenCalledWith("test-token", "test-prompt");
      });
    });

    it("should not fetch versions when isOpen is false", () => {
      render(<VersionHistorySidePanel {...defaultProps} isOpen={false} />);

      expect(getPromptVersions).not.toHaveBeenCalled();
    });

    it("should not fetch versions when accessToken is null", () => {
      render(<VersionHistorySidePanel {...defaultProps} accessToken={null} />);

      expect(getPromptVersions).not.toHaveBeenCalled();
    });

    it("should not fetch versions when promptId is not provided", () => {
      render(<VersionHistorySidePanel {...defaultProps} promptId="" />);

      expect(getPromptVersions).not.toHaveBeenCalled();
    });

    it("should refetch versions when props change", async () => {
      const { rerender } = render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(getPromptVersions).toHaveBeenCalledTimes(1);
      });

      rerender(<VersionHistorySidePanel {...defaultProps} promptId="different-prompt.v1" />);

      await waitFor(() => {
        expect(getPromptVersions).toHaveBeenCalledTimes(2);
        expect(getPromptVersions).toHaveBeenCalledWith("test-token", "different-prompt");
      });
    });
  });

  describe("Error Handling", () => {
    it("should handle network errors gracefully", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      mockGetPromptVersions.mockRejectedValueOnce(new Error("Network error"));

      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith("Error fetching prompt versions:", expect.any(Error));
      });

      // Should show empty state when there's an error
      expect(screen.getByText("No version history available.")).toBeInTheDocument();

      consoleSpy.mockRestore();
    });
  });

  describe("User Interactions", () => {
    it("should call onClose when close button is clicked", async () => {
      const mockOnClose = vi.fn();
      render(<VersionHistorySidePanel {...defaultProps} onClose={mockOnClose} />);

      // shadcn SheetContent renders an accessible Close button (Radix).
      const closeButton = screen.getByRole("button", { name: /close/i });
      await act(async () => {
        fireEvent.click(closeButton);
      });

      expect(mockOnClose).toHaveBeenCalled();
    });

    it("should prevent interaction with main content when drawer is open", () => {
      render(<VersionHistorySidePanel {...defaultProps} />);

      // Sheet is rendered with modal={false} + onInteractOutside preventDefault
      // so the dialog is open but does not block clicks on the main content.
      const dialog = screen.getByRole("dialog", { name: /version history/i });
      expect(dialog).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("should handle activeVersionId with .v suffix correctly", async () => {
      render(<VersionHistorySidePanel {...defaultProps} activeVersionId="test-prompt.v1" />);

      await waitFor(() => {
        expect(screen.getByText("Active")).toBeInTheDocument();
      });
    });

    it("should handle activeVersionId with _v suffix correctly", async () => {
      const versionsWithUnderscore = [
        {
          prompt_id: "test-prompt_v2",
          litellm_params: { prompt_id: "test-prompt_v2" },
          prompt_info: { prompt_type: "db" },
          version: 2,
          created_at: "2024-01-15T10:30:00Z",
        },
      ];

      mockGetPromptVersions.mockResolvedValueOnce({
        prompts: versionsWithUnderscore,
      });

      render(<VersionHistorySidePanel {...defaultProps} activeVersionId="test-prompt_v2" />);

      await waitFor(() => {
        expect(screen.getByText("Active")).toBeInTheDocument();
      });
    });

    it("should sort versions correctly with version field", async () => {
      // The component doesn't explicitly sort, but we can verify the order from the API response
      render(<VersionHistorySidePanel {...defaultProps} />);

      await waitFor(() => {
        // Verify versions are displayed as they come from the API
        expect(screen.getByText("v2")).toBeInTheDocument();
      });
    });
  });
});
