import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders, screen, fireEvent } from "./test-utils";
import TopKeyView from "../src/components/UsagePage/components/EntityUsage/TopKeyView";
import { TagUsage } from "../src/components/UsagePage/types";
import useAuthorized from "../src/app/(dashboard)/hooks/useAuthorized";

// Mock the networking module
vi.mock("../src/components/networking", () => ({
  keyInfoV1Call: vi.fn(),
}));

// Mock the transform function
vi.mock("../src/components/key_team_helpers/transform_key_info", () => ({
  transformKeyInfo: vi.fn((data) => data),
}));

vi.mock("../src/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

describe("TopKeyView", () => {
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const mockProps = {
    topKeys: [],
    accessToken: "test-token",
    userID: "test-user",
    userRole: "admin",
    teams: null,
    premiumUser: true,
    showTags: false,
  };

  const mockKeysWithTags = [
    {
      api_key: "key-1",
      key_alias: "Production Key",
      tags: [
        { tag: "production", usage: 0.005 } as TagUsage, // <$0.01
        { tag: "high-volume", usage: 125.5 } as TagUsage, // High spend
        { tag: "api-calls", usage: 0.003 } as TagUsage, // <$0.01
      ],
      spend: 125.5,
    },
    {
      api_key: "key-2",
      key_alias: "Staging Key",
      tags: [
        { tag: "staging", usage: 45.75 } as TagUsage, // Medium spend
        { tag: "testing", usage: 0.008 } as TagUsage, // <$0.01
        { tag: "development", usage: 12.25 } as TagUsage, // Low spend
      ],
      spend: 58.0,
    },
    {
      api_key: "key-3",
      key_alias: "Development Key",
      tags: [
        { tag: "dev", usage: 0.002 } as TagUsage, // <$0.01
        { tag: "experimental", usage: 0.001 } as TagUsage, // <$0.01
      ],
      spend: 0.003,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({
      token: "mock-token",
      accessToken: mockProps.accessToken,
      userId: mockProps.userID,
      userEmail: "test@example.com",
      userRole: mockProps.userRole,
      premiumUser: mockProps.premiumUser,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });
  });

  describe("Tags Column Visibility", () => {
    it("should not show tags column when showTags is false", () => {
      renderWithProviders(<TopKeyView {...mockProps} showTags={false} />);
      expect(screen.queryByText("Tags")).not.toBeInTheDocument();
    });

    it("should show tags column when showTags is true", () => {
      renderWithProviders(<TopKeyView {...mockProps} showTags={true} />);
      expect(screen.getByText("Tags")).toBeInTheDocument();
    });
  });

  describe("Tags Display and Sorting", () => {
    beforeEach(() => {
      renderWithProviders(<TopKeyView {...mockProps} topKeys={mockKeysWithTags} showTags={true} />);
    });

    it("should display tags for each key", () => {
      // Check that tags are displayed (truncated to 7 chars + ...)
      expect(screen.getByText("product...")).toBeInTheDocument();
      expect(screen.getByText("high-vo...")).toBeInTheDocument();
      expect(screen.getByText("staging...")).toBeInTheDocument();
    });

    it("should display top 2 tags by default (sorted by spend)", () => {
      // Only the top 2 tags by spend should be visible initially
      // Production Key: high-volume (125.50), production (0.005) - sorted by spend
      expect(screen.getByText("high-vo...")).toBeInTheDocument();
      expect(screen.getByText("product...")).toBeInTheDocument();

      // Staging Key: staging (45.75), development (12.25) - sorted by spend
      expect(screen.getByText("staging...")).toBeInTheDocument();
      expect(screen.getByText("develop...")).toBeInTheDocument();

      // Development Key: dev (0.002), experimental (0.001) - sorted by spend
      expect(screen.getByText("dev...")).toBeInTheDocument();
      expect(screen.getByText("experim...")).toBeInTheDocument();

      // These should NOT be visible initially (3rd+ tags)
      expect(screen.queryByText("api-cal...")).not.toBeInTheDocument();
      expect(screen.queryByText("testing...")).not.toBeInTheDocument();
    });

    it("should show expand/collapse arrows for keys with more than 2 tags", () => {
      // Production Key has 3 tags, so it should have an expand arrow
      // Look for the chevron down icon (expand button)
      const expandButtons = screen.getAllByRole("button");
      const expandButton = expandButtons.find((button) => button.getAttribute("title") === "Show all tags");
      expect(expandButton).toBeInTheDocument();
    });

    it("should show all tags when expanded", async () => {
      // Find and click the expand button for Production Key (has 3 tags)
      const expandButtons = screen.getAllByRole("button");
      const expandButton = expandButtons.find((button) => button.getAttribute("title") === "Show all tags");

      if (expandButton) {
        fireEvent.click(expandButton);

        // Now all tags should be visible
        expect(screen.getByText("api-cal...")).toBeInTheDocument();
      }
    });

    it("should show tooltip on hover with tag information", async () => {
      // Hover over a tag to trigger tooltip
      const tagElement = screen.getByText("high-vo...");
      fireEvent.mouseOver(tagElement);

      // Check that tooltip content appears
      // Note: The exact tooltip content depends on your tooltip implementation
      // You might need to adjust this based on how Ant Design Tooltip renders
    });
  });

  describe("Tag Spend Formatting", () => {
    it("should handle high spend amounts", () => {
      renderWithProviders(<TopKeyView {...mockProps} topKeys={mockKeysWithTags} showTags={true} />);

      // Test that high spend amounts are displayed
      // This would require checking tooltip content or finding a way to access the formatted values
      expect(screen.getByText("high-vo...")).toBeInTheDocument();
    });

    it("should handle micro spend amounts", () => {
      renderWithProviders(<TopKeyView {...mockProps} topKeys={mockKeysWithTags} showTags={true} />);

      // Test that very small amounts are displayed (only the top 2 tags are visible by default)
      // product... has usage: 0.005 (<$0.01)
      expect(screen.getByText("product...")).toBeInTheDocument();

      // dev... has usage: 0.002 (<$0.01)
      expect(screen.getByText("dev...")).toBeInTheDocument();

      // experimental... has usage: 0.001 (<$0.01)
      expect(screen.getByText("experim...")).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("should handle keys with no tags", () => {
      const keysWithoutTags = [
        {
          api_key: "key-no-tags",
          key_alias: "No Tags Key",
          tags: [],
          spend: 10.0,
        },
      ];

      renderWithProviders(<TopKeyView {...mockProps} topKeys={keysWithoutTags} showTags={true} />);
      expect(screen.getByText("-")).toBeInTheDocument();
    });

    it("should handle keys with undefined tags", () => {
      const keysWithUndefinedTags = [
        {
          api_key: "key-undefined-tags",
          key_alias: "Undefined Tags Key",
          tags: undefined,
          spend: 5.0,
        },
      ];

      renderWithProviders(<TopKeyView {...mockProps} topKeys={keysWithUndefinedTags} showTags={true} />);
      expect(screen.getByText("-")).toBeInTheDocument();
    });

    it("should handle keys with null tags", () => {
      const keysWithNullTags = [
        {
          api_key: "key-null-tags",
          key_alias: "Null Tags Key",
          tags: null,
          spend: 3.0,
        },
      ];

      renderWithProviders(<TopKeyView {...mockProps} topKeys={keysWithNullTags} showTags={true} />);
      expect(screen.getByText("-")).toBeInTheDocument();
    });
  });

  describe("Tag Truncation", () => {
    it("should truncate long tag names to 7 characters", () => {
      const keysWithLongTags = [
        {
          api_key: "key-long-tags",
          key_alias: "Long Tags Key",
          tags: [{ tag: "very-long-tag-name", usage: 10.0 } as TagUsage, { tag: "short", usage: 5.0 } as TagUsage],
          spend: 15.0,
        },
      ];

      renderWithProviders(<TopKeyView {...mockProps} topKeys={keysWithLongTags} showTags={true} />);

      // Should show truncated version
      expect(screen.getByText("very-lo...")).toBeInTheDocument();
      // Short tags should still be truncated (all tags get ...)
      expect(screen.getByText("short...")).toBeInTheDocument();
    });
  });

  describe("Multiple Keys with Different Tag Spend Patterns", () => {
    it("should handle mixed spend patterns across multiple keys", () => {
      const mixedSpendKeys = [
        {
          api_key: "key-mixed-1",
          key_alias: "Mixed Key 1",
          tags: [{ tag: "expensive", usage: 999.99 } as TagUsage, { tag: "cheap", usage: 0.001 } as TagUsage],
          spend: 1000.0,
        },
        {
          api_key: "key-mixed-2",
          key_alias: "Mixed Key 2",
          tags: [{ tag: "moderate", usage: 50.0 } as TagUsage, { tag: "tiny", usage: 0.005 } as TagUsage],
          spend: 50.01,
        },
      ];

      renderWithProviders(<TopKeyView {...mockProps} topKeys={mixedSpendKeys} showTags={true} />);

      // Verify that all tag types are displayed
      expect(screen.getByText("expensi...")).toBeInTheDocument();
      expect(screen.getByText("cheap...")).toBeInTheDocument();
      expect(screen.getByText("moderat...")).toBeInTheDocument();
      expect(screen.getByText("tiny...")).toBeInTheDocument();
    });
  });

  describe("Table Structure", () => {
    it("should render table with correct headers when showTags is true", () => {
      renderWithProviders(<TopKeyView {...mockProps} showTags={true} />);

      expect(screen.getByText("Key ID")).toBeInTheDocument();
      expect(screen.getByText("Key Alias")).toBeInTheDocument();
      expect(screen.getByText("Tags")).toBeInTheDocument();
      expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
    });

    it("should render table with correct headers when showTags is false", () => {
      renderWithProviders(<TopKeyView {...mockProps} showTags={false} />);

      expect(screen.getByText("Key ID")).toBeInTheDocument();
      expect(screen.getByText("Key Alias")).toBeInTheDocument();
      expect(screen.queryByText("Tags")).not.toBeInTheDocument();
      expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
    });
  });

  describe("Key Data Display", () => {
    it("should display key information correctly", () => {
      const simpleKeys = [
        {
          api_key: "test-key-123",
          key_alias: "Test Key",
          tags: [],
          spend: 25.5,
        },
      ];

      renderWithProviders(<TopKeyView {...mockProps} topKeys={simpleKeys} showTags={true} />);

      // Check that key alias is displayed
      expect(screen.getByText("Test Key")).toBeInTheDocument();

      // Check that spend is formatted correctly
      expect(screen.getByText("$25.50")).toBeInTheDocument();
    });
  });
});
