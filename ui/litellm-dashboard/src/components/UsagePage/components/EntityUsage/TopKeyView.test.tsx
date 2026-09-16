import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { KeyResponse } from "../../../key_team_helpers/key_list";
import * as networking from "../../../networking";
import TopKeyView from "./TopKeyView";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: vi.fn(),
}));

vi.mock("../../../networking", () => ({
  keyInfoV1Call: vi.fn(),
}));

vi.mock("../../../templates/key_info_view", () => ({
  default: ({
    keyId,
    onClose,
    keyData,
    onKeyDataUpdate,
  }: {
    keyId: string;
    onClose: () => void;
    keyData: KeyResponse | undefined;
    onKeyDataUpdate?: (data: Partial<KeyResponse>) => void;
  }) => (
    <div data-testid="key-info-view">
      <div>Key Info View for {keyId}</div>
      <div>{keyData ? `Loaded alias ${keyData.key_alias}` : "Key not found"}</div>
      <button onClick={onClose}>Close</button>
      <button onClick={() => onKeyDataUpdate?.({ token: "rotated-hash" })}>Rotate</button>
      <button onClick={() => onKeyDataUpdate?.({ spend: 0 })}>Reset spend</button>
    </div>
  ),
}));

type UrlUpdateMock = ReturnType<typeof vi.fn<OnUrlUpdateFunction>>;

const lastUrlUpdate = (onUrlUpdate: UrlUpdateMock) => onUrlUpdate.mock.calls.at(-1)?.[0];

describe("TopKeyView", () => {
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const mockKeyInfoV1Call = vi.mocked(networking.keyInfoV1Call);

  const mockAuth: ReturnType<typeof useAuthorized> = {
    isLoading: false,
    isAuthorized: true,
    token: "mock-token",
    accessToken: "test-token",
    userId: "user-1",
    userEmail: "user@example.com",
    userRole: "admin",
    userRoleLabel: "Admin",
    isViewOnly: false,
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

  const oneKey = [{ api_key: "key-123", key_alias: "Test Key", spend: 100 }];

  beforeEach(() => {
    testQueryClient.clear();
    mockUseAuthorized.mockReturnValue(mockAuth);
    mockSetTopKeysLimit.mockClear();
    mockKeyInfoV1Call.mockReset();
    mockKeyInfoV1Call.mockResolvedValue({ key: "key-123", info: { key_alias: "Fetched Alias" } });
  });

  it("should render", () => {
    renderWithProviders(<TopKeyView {...baseProps} />);
    expect(screen.getByRole("button", { name: "Table View" })).toBeInTheDocument();
  });

  it("should display table view button", () => {
    renderWithProviders(<TopKeyView {...baseProps} />);
    expect(screen.getByRole("button", { name: "Table View" })).toBeInTheDocument();
  });

  it("should display chart view button", () => {
    renderWithProviders(<TopKeyView {...baseProps} />);
    expect(screen.getByRole("button", { name: "Chart View" })).toBeInTheDocument();
  });

  it("should display base table column headers", () => {
    renderWithProviders(<TopKeyView {...baseProps} />);
    expect(screen.getByText("Key ID")).toBeInTheDocument();
    expect(screen.getByText("Key Alias")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
  });

  it("should display Tags column when showTags is true", () => {
    renderWithProviders(<TopKeyView {...baseProps} showTags={true} />);
    expect(screen.getByText("Tags")).toBeInTheDocument();
  });

  it("should not display Tags column when showTags is false", () => {
    renderWithProviders(<TopKeyView {...baseProps} showTags={false} />);
    expect(screen.queryByText("Tags")).not.toBeInTheDocument();
  });

  it("should display key information in table view", () => {
    renderWithProviders(
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
    renderWithProviders(<TopKeyView {...baseProps} />);

    const chartViewButton = screen.getByRole("button", { name: "Chart View" });
    await user.click(chartViewButton);

    expect(chartViewButton).toHaveClass("bg-info/15");
  });

  it("renders cyan bars with truncated aliases in chart view and opens the key info modal on bar click", async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "A Very Long Key Alias",
            spend: 100,
          },
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Chart View" }));

    const bars = container.querySelectorAll("path.recharts-rectangle");
    expect(bars).toHaveLength(1);
    expect(bars[0]).toHaveAttribute("fill", "var(--color-cyan-500, #06b6d4)");
    expect(screen.getAllByText("A Very Lon...").length).toBeGreaterThan(0);

    fireEvent.click(bars[0]);

    await waitFor(() => {
      expect(mockKeyInfoV1Call).toHaveBeenCalledWith("test-token", "key-123");
    });

    await waitFor(() => {
      expect(screen.getByTestId("key-info-view")).toBeInTheDocument();
    });
    expect(screen.getByText("Key Info View for key-123")).toBeInTheDocument();
    expect(screen.getByText("Loaded alias Fetched Alias")).toBeInTheDocument();
  });

  it("should switch to table view when table view button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TopKeyView {...baseProps} />);

    const chartViewButton = screen.getByRole("button", { name: "Chart View" });
    const tableViewButton = screen.getByRole("button", { name: "Table View" });

    await user.click(chartViewButton);
    await user.click(tableViewButton);

    expect(tableViewButton).toHaveClass("bg-info/15");
  });

  it("should call setTopKeysLimit when limit is changed via the segmented control", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TopKeyView {...baseProps} />);

    await user.click(screen.getByRole("radio", { name: "10" }));

    expect(mockSetTopKeysLimit).toHaveBeenCalledWith(10);
  });

  it("should display truncated key ID in table", () => {
    renderWithProviders(
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
    const keyId = screen.getByText("sk-1234567890abcdef");
    expect(keyId).toBeInTheDocument();
    expect(keyId).toHaveClass("truncate");
  });

  it("should display dash for missing key alias", () => {
    renderWithProviders(
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
    renderWithProviders(
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

  it("should display sub-cent spend as < $0.01", () => {
    renderWithProviders(
      <TopKeyView
        {...baseProps}
        topKeys={[
          {
            api_key: "key-123",
            key_alias: "Test Key",
            spend: 0.004,
          },
        ]}
      />,
    );
    expect(screen.getByText("< $0.01")).toBeInTheDocument();
  });

  it("should display zero spend as a dash", () => {
    renderWithProviders(
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
    expect(screen.getByText("-")).toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
  });

  it("should display dash for empty tags", () => {
    renderWithProviders(
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
    renderWithProviders(
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
    renderWithProviders(
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
    renderWithProviders(
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
    renderWithProviders(
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
    const user = userEvent.setup();
    renderWithProviders(
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

    const keyIdButton = screen.getByText("key-123").closest("button");
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
    const user = userEvent.setup();
    renderWithProviders(
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

    const keyIdButton = screen.getByText("key-123").closest("button");
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
    const user = userEvent.setup();
    renderWithProviders(
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

    const keyIdButton = screen.getByText("key-123").closest("button");
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
    const user = userEvent.setup();
    const { container } = renderWithProviders(
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

    const keyIdButton = screen.getByText("key-123").closest("button");
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

    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const user = userEvent.setup();
    renderWithProviders(
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
      { onUrlUpdate },
    );

    const keyIdButton = screen.getByText("key-123").closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(mockKeyInfoV1Call).not.toHaveBeenCalled();
    });

    expect(screen.queryByTestId("key-info-view")).not.toBeInTheDocument();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  it("should show the not-found state when fetching key info fails", async () => {
    mockKeyInfoV1Call.mockRejectedValue(new Error("Network error"));

    const user = userEvent.setup();
    renderWithProviders(
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

    const keyIdButton = screen.getByText("key-123").closest("button");
    if (keyIdButton) {
      await user.click(keyIdButton);
    }

    await waitFor(() => {
      expect(mockKeyInfoV1Call).toHaveBeenCalled();
    });

    expect(await screen.findByText("Key not found")).toBeInTheDocument();
    expect(screen.getByText("Key Info View for key-123")).toBeInTheDocument();
  });

  it("should sort tags by usage descending", () => {
    renderWithProviders(
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
    renderWithProviders(<TopKeyView {...baseProps} topKeys={[]} />);
    expect(screen.getByText("Key ID")).toBeInTheDocument();
    expect(screen.getByText("Key Alias")).toBeInTheDocument();
  });

  it("should display full key alias in table view", () => {
    renderWithProviders(
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

  describe("URL state", () => {
    const openFirstKey = async (user: ReturnType<typeof userEvent.setup>) => {
      const keyIdButton = screen.getByText("key-123").closest("button");
      expect(keyIdButton).not.toBeNull();
      await user.click(keyIdButton as HTMLButtonElement);
    };

    it("opens the key detail named by ?key= without a click", async () => {
      renderWithProviders(<TopKeyView {...baseProps} topKeys={oneKey} />, { searchParams: "?key=key-123" });

      expect(screen.getByText("Loading key...")).toBeInTheDocument();
      expect(await screen.findByText("Loaded alias Fetched Alias")).toBeInTheDocument();
      expect(screen.getByText("Key Info View for key-123")).toBeInTheDocument();
      expect(mockKeyInfoV1Call).toHaveBeenCalledWith("test-token", "key-123");
    });

    it("pushes ?key= when a key is clicked so Back closes the detail", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<TopKeyView {...baseProps} topKeys={oneKey} />, { onUrlUpdate });

      await openFirstKey(user);

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("key")).toBe("key-123"));
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    });

    it.each([
      [
        "the close button",
        async (user: ReturnType<typeof userEvent.setup>) => user.click(screen.getByLabelText("Close")),
      ],
      ["Escape", async (user: ReturnType<typeof userEvent.setup>) => user.keyboard("{Escape}")],
    ])("drops ?key= when the detail is closed with %s", async (_label, close) => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<TopKeyView {...baseProps} topKeys={oneKey} />, {
        searchParams: "?key=key-123&view=team",
        onUrlUpdate,
      });
      await screen.findByText("Loaded alias Fetched Alias");

      await close(user);

      await waitFor(() => expect(screen.queryByTestId("key-info-view")).not.toBeInTheDocument());
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("key")).toBe(false);
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("view")).toBe("team");
    });

    it("repoints ?key= at the rotated hash without adding a history entry", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<TopKeyView {...baseProps} topKeys={oneKey} />, {
        searchParams: "?key=key-123",
        onUrlUpdate,
      });
      await screen.findByText("Loaded alias Fetched Alias");

      await user.click(screen.getByRole("button", { name: "Reset spend" }));
      await user.click(screen.getByRole("button", { name: "Rotate" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("key")).toBe("rotated-hash"));
      expect(onUrlUpdate).toHaveBeenCalledTimes(1);
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
      expect(await screen.findByText("Key Info View for rotated-hash")).toBeInTheDocument();
      expect(mockKeyInfoV1Call).toHaveBeenLastCalledWith("test-token", "rotated-hash");
    });

    it("shows the view mode named by ?top_keys_view=", () => {
      renderWithProviders(<TopKeyView {...baseProps} topKeys={oneKey} />, { searchParams: "?top_keys_view=chart" });

      expect(screen.getByRole("button", { name: "Chart View" })).toHaveClass("bg-info/15");
      expect(screen.getByRole("button", { name: "Table View" })).not.toHaveClass("bg-info/15");
      expect(screen.queryByText("Key Alias")).not.toBeInTheDocument();
    });

    it("falls back to the table when ?top_keys_view= is unknown", () => {
      renderWithProviders(<TopKeyView {...baseProps} topKeys={oneKey} />, { searchParams: "?top_keys_view=pie" });

      expect(screen.getByRole("button", { name: "Table View" })).toHaveClass("bg-info/15");
      expect(screen.getByText("Key Alias")).toBeInTheDocument();
    });

    it("writes ?top_keys_view=chart and drops it again for the default table", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<TopKeyView {...baseProps} topKeys={oneKey} />, { onUrlUpdate });

      await user.click(screen.getByRole("button", { name: "Chart View" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("top_keys_view")).toBe("chart"));
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");

      await user.click(screen.getByRole("button", { name: "Table View" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("top_keys_view")).toBe(false));
    });
  });

  it("should handle keys with no alias", () => {
    renderWithProviders(
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
