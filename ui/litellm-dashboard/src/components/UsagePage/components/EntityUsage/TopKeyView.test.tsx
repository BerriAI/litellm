import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { KeyResponse } from "../../../key_team_helpers/key_list";
import * as transformKeyInfo from "../../../key_team_helpers/transform_key_info";
import * as networking from "../../../networking";
import TopKeyView from "./TopKeyView";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: vi.fn(),
}));

vi.mock("../../../networking", () => ({
  keyInfoV1Call: vi.fn(),
}));

vi.mock("../../../key_team_helpers/transform_key_info", () => ({
  transformKeyInfo: vi.fn(),
}));

vi.mock("../../../templates/key_info_view", () => ({
  default: ({ keyId, onClose }: { keyId: string; onClose: () => void }) => (
    <div data-testid="key-info-view">
      <div>Key Info View for {keyId}</div>
      <button onClick={onClose}>Close</button>
    </div>
  ),
}));

describe("TopKeyView", () => {
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const mockKeyInfoV1Call = vi.mocked(networking.keyInfoV1Call);
  const mockTransformKeyInfo = vi.mocked(transformKeyInfo.transformKeyInfo);

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

  const mockSetTopKeysLimit = vi.fn();

  const baseProps = {
    topKeys: [],
    teams: null,
    showTags: false,
    topKeysLimit: 5,
    setTopKeysLimit: mockSetTopKeysLimit,
  };

  beforeEach(() => {
    mockUseAuthorized.mockReturnValue(mockAuth);
    mockSetTopKeysLimit.mockClear();
    mockKeyInfoV1Call.mockClear();
    mockTransformKeyInfo.mockClear();
  });

  it("should render", () => {
    render(<TopKeyView {...baseProps} />);
    expect(screen.getByRole("button", { name: "Table View" })).toBeInTheDocument();
  });

  it("should display table view button", () => {
    render(<TopKeyView {...baseProps} />);
    expect(screen.getByRole("button", { name: "Table View" })).toBeInTheDocument();
  });

  it("should display chart view button", () => {
    render(<TopKeyView {...baseProps} />);
    expect(screen.getByRole("button", { name: "Chart View" })).toBeInTheDocument();
  });

  it("should display base table column headers", () => {
    render(<TopKeyView {...baseProps} />);
    expect(screen.getByText("Key ID")).toBeInTheDocument();
    expect(screen.getByText("Key Alias")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
  });

  it("should display Tags column when showTags is true", () => {
    render(<TopKeyView {...baseProps} showTags={true} />);
    expect(screen.getByText("Tags")).toBeInTheDocument();
  });

  it("should not display Tags column when showTags is false", () => {
    render(<TopKeyView {...baseProps} showTags={false} />);
    expect(screen.queryByText("Tags")).not.toBeInTheDocument();
  });

  it("should display key information in table view", () => {
    render(
      <TopKeyView
        {...baseProps}
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
        showTags={true}
      />,
    );
    expect(screen.getByText("Test Key")).toBeInTheDocument();
    expect(screen.getByText(/tag-1/)).toBeInTheDocument();
    expect(screen.getByText(/tag-2/)).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
  });

  it("should switch to chart view when chart view button is clicked", async () => {
    const user = userEvent.setup();
    render(<TopKeyView {...baseProps} />);

    const chartViewButton = screen.getByRole("button", { name: "Chart View" });
    await user.click(chartViewButton);

    expect(chartViewButton).toHaveClass("bg-blue-100");
  });

  it("should switch to table view when table view button is clicked", async () => {
    const user = userEvent.setup();
    render(<TopKeyView {...baseProps} />);

    const chartViewButton = screen.getByRole("button", { name: "Chart View" });
    const tableViewButton = screen.getByRole("button", { name: "Table View" });

    await user.click(chartViewButton);
    await user.click(tableViewButton);

    expect(tableViewButton).toHaveClass("bg-blue-100");
  });

  it("should call setTopKeysLimit when limit is changed via Segmented control", async () => {
    const user = userEvent.setup();
    render(<TopKeyView {...baseProps} />);

    const limit10Radio = screen.getByRole("radio", { name: "10" });
    const limit10Label = limit10Radio.closest("label");
    if (limit10Label) {
      await user.click(limit10Label);
    } else {
      // Fallback: click the div with title="10"
      const limit10Div = screen.getByTitle("10");
      await user.click(limit10Div);
    }

    expect(mockSetTopKeysLimit).toHaveBeenCalledWith(10);
  });

  it("should display truncated key ID in table", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "sk-1234567890abcdef",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
      />,
    );
    expect(screen.getByText(/sk-1234\.\.\./)).toBeInTheDocument();
  });

  it("should display dash for missing key alias", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "",
            spend: 100,
          },
        ]}
      />,
    );
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should format spend values with two decimal places", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 123.456,
          },
        ]}
      />,
    );
    expect(screen.getByText("$123.46")).toBeInTheDocument();
  });

  it("should display less than 0.01 spend as <$0.01", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 0.005,
          },
        ]}
      />,
    );
    expect(screen.getByText("<$0.01")).toBeInTheDocument();
  });

  it("should display zero spend correctly", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 0,
          },
        ]}
      />,
    );
    expect(screen.getByText("$0.00")).toBeInTheDocument();
  });

  it("should display dash for empty tags", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
            tags: [],
          },
        ]}
        showTags={true}
      />,
    );
    expect(screen.getAllByText("-").length).toBeGreaterThan(0);
  });

  it("should display dash for missing tags", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
        showTags={true}
      />,
    );
    expect(screen.getAllByText("-").length).toBeGreaterThan(0);
  });

  it("should display first two tags by default and show expand button", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
            tags: [
              { tag: "tag-1", usage: 50 },
              { tag: "tag-2", usage: 30 },
              { tag: "tag-3", usage: 20 },
            ],
          },
        ]}
        showTags={true}
      />,
    );
    expect(screen.getByText(/tag-1/)).toBeInTheDocument();
    expect(screen.getByText(/tag-2/)).toBeInTheDocument();
    expect(screen.queryByText(/tag-3/)).not.toBeInTheDocument();
  });

  it("should expand tags when expand button is clicked", async () => {
    const user = userEvent.setup();
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
            tags: [
              { tag: "tag-1", usage: 50 },
              { tag: "tag-2", usage: 30 },
              { tag: "tag-3", usage: 20 },
            ],
          },
        ]}
        showTags={true}
      />,
    );

    const expandButton = screen.getByTitle("Show all tags");
    await user.click(expandButton);

    expect(screen.getByText(/tag-3/)).toBeInTheDocument();
  });

  it("should collapse tags when collapse button is clicked", async () => {
    const user = userEvent.setup();
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
            tags: [
              { tag: "tag-1", usage: 50 },
              { tag: "tag-2", usage: 30 },
              { tag: "tag-3", usage: 20 },
            ],
          },
        ]}
        showTags={true}
      />,
    );

    const expandButton = screen.getByTitle("Show all tags");
    await user.click(expandButton);

    expect(screen.getByText(/tag-3/)).toBeInTheDocument();

    const collapseButton = screen.getByTitle("Show fewer tags");
    await user.click(collapseButton);

    expect(screen.queryByText(/tag-3/)).not.toBeInTheDocument();
  });

  it("should open modal when key ID is clicked", async () => {
    const mockKeyInfo = { key: "info" };
    const mockTransformedData = { transformed: "data" } as unknown as KeyResponse;
    mockKeyInfoV1Call.mockResolvedValue(mockKeyInfo);
    mockTransformKeyInfo.mockReturnValue(mockTransformedData);

    const user = userEvent.setup();
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
      />,
    );

    const keyIdButton = screen.getByText(/key-123\.\.\./).closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(mockKeyInfoV1Call).toHaveBeenCalledWith("test-token", "key-123");
    });

    await waitFor(() => {
      expect(screen.getByTestId("key-info-view")).toBeInTheDocument();
    });
  });

  it("should close modal when close button is clicked", async () => {
    const mockKeyInfo = { key: "info" };
    const mockTransformedData = { transformed: "data" } as unknown as KeyResponse;
    mockKeyInfoV1Call.mockResolvedValue(mockKeyInfo);
    mockTransformKeyInfo.mockReturnValue(mockTransformedData);

    const user = userEvent.setup();
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
      />,
    );

    const keyIdButton = screen.getByText(/key-123\.\.\./).closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(screen.getByTestId("key-info-view")).toBeInTheDocument();
    });

    const closeButton = screen.getByLabelText("Close");
    await user.click(closeButton);

    await waitFor(() => {
      expect(screen.queryByTestId("key-info-view")).not.toBeInTheDocument();
    });
  });

  it("should close modal when escape key is pressed", async () => {
    const mockKeyInfo = { key: "info" };
    const mockTransformedData = { transformed: "data" } as unknown as KeyResponse;
    mockKeyInfoV1Call.mockResolvedValue(mockKeyInfo);
    mockTransformKeyInfo.mockReturnValue(mockTransformedData);

    const user = userEvent.setup();
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
      />,
    );

    const keyIdButton = screen.getByText(/key-123\.\.\./).closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(screen.getByTestId("key-info-view")).toBeInTheDocument();
    });

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByTestId("key-info-view")).not.toBeInTheDocument();
    });
  });

  it("should close modal when clicking outside modal", async () => {
    const mockKeyInfo = { key: "info" };
    const mockTransformedData = { transformed: "data" } as unknown as KeyResponse;
    mockKeyInfoV1Call.mockResolvedValue(mockKeyInfo);
    mockTransformKeyInfo.mockReturnValue(mockTransformedData);

    const user = userEvent.setup();
    const { container } = render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
      />,
    );

    const keyIdButton = screen.getByText(/key-123\.\.\./).closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(screen.getByTestId("key-info-view")).toBeInTheDocument();
    });

    const modalBackdrop = container.querySelector(".fixed.inset-0");
    if (modalBackdrop) {
      await user.click(modalBackdrop);
    }

    await waitFor(() => {
      expect(screen.queryByTestId("key-info-view")).not.toBeInTheDocument();
    });
  });

  it("should not open modal when accessToken is missing", async () => {
    mockUseAuthorized.mockReturnValue({
      ...mockAuth,
      accessToken: "",
    });

    const user = userEvent.setup();
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
      />,
    );

    const keyIdButton = screen.getByText(/key-123\.\.\./).closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(mockKeyInfoV1Call).not.toHaveBeenCalled();
    });

    expect(screen.queryByTestId("key-info-view")).not.toBeInTheDocument();
  });

  it("should handle error when fetching key info", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mockKeyInfoV1Call.mockRejectedValue(new Error("Network error"));

    const user = userEvent.setup();
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
          },
        ]}
      />,
    );

    const keyIdButton = screen.getByText(/key-123\.\.\./).closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(mockKeyInfoV1Call).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(consoleErrorSpy).toHaveBeenCalled();
    });

    consoleErrorSpy.mockRestore();
  });

  it("should sort tags by usage descending", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 100,
            tags: [
              { tag: "tag-low", usage: 10 },
              { tag: "tag-high", usage: 50 },
              { tag: "tag-medium", usage: 30 },
            ],
          },
        ]}
        showTags={true}
      />,
    );

    const tagElements = screen.getAllByText(/tag-/);
    const tagTexts = tagElements.map((el) => el.textContent);
    // Tags are truncated to 7 characters + "...", so "tag-high" becomes "tag-hig..."
    expect(tagTexts[0]).toMatch(/^tag-hig/);
    expect(tagTexts[1]).toMatch(/^tag-med/);
  });

  it("should handle empty key list", () => {
    render(<TopKeyView {...baseProps} topKeys={[]} />);
    expect(screen.getByText("Key ID")).toBeInTheDocument();
    expect(screen.getByText("Key Alias")).toBeInTheDocument();
  });

  it("should display full key alias in table view", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "This is a very long key alias",
            spend: 100,
          },
        ]}
      />,
    );
    expect(screen.getByText("This is a very long key alias")).toBeInTheDocument();
  });

  it("should handle keys with no alias", () => {
    render(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: null,
            spend: 100,
          },
        ]}
      />,
    );
    expect(screen.getByText("-")).toBeInTheDocument();
  });
});
