import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import TopKeyView from "./TopKeyView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: vi.fn(),
}));

describe("TopKeyView", () => {
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const mockAuth = {
    token: "mock-token",
    accessToken: "test-token",
    userId: "user-1",
    userEmail: "user@example.com",
    userRole: "admin",
    premiumUser: true,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  };
  const baseProps = {
    topKeys: [],
    teams: null,
    showTags: false,
  };

  beforeEach(() => {
    mockUseAuthorized.mockReturnValue(mockAuth);
  });

  it("should render", () => {
    render(<TopKeyView {...baseProps} />);
    expect(screen.getByText("Table View")).toBeInTheDocument();
  });

  it("should have a table view button", () => {
    render(<TopKeyView {...baseProps} />);
    expect(screen.getByText("Table View")).toBeInTheDocument();
  });

  it("should have a chart view", () => {
    render(<TopKeyView {...baseProps} />);
    expect(screen.getByText("Chart View")).toBeInTheDocument();
  });

  ["Key ID", "Key Alias", "Spend (USD)"].forEach((header) => {
    it(`should have a ${header} column`, () => {
      render(<TopKeyView {...baseProps} />);
      expect(screen.getByText(header)).toBeInTheDocument();
    });
  });

  it("should have a Tags column when showTags is true", () => {
    render(<TopKeyView {...baseProps} showTags={true} />);
    expect(screen.getByText("Tags")).toBeInTheDocument();
  });

  it("should show the key's information on the table", () => {
    render(
      <TopKeyView
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
            tags: [
              { tag: "tag-1", usage: 50 },
              { tag: "tag-2", usage: 30 },
            ],
          },
        ]}
        teams={null}
        showTags={true}
      />,
    );
    expect(screen.getByText("Test Key")).toBeInTheDocument();
    expect(screen.getByText(/tag-1/)).toBeInTheDocument();
    expect(screen.getByText(/tag-2/)).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
  });
});
