import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { tagDeleteCall, tagListCall } from "@/components/networking";
import type { Tag } from "@/components/tag_management/types";

import { renderWithProviders } from "@/../tests/test-utils";
import TagManagement from "./index";

vi.mock("@/components/networking", () => ({
  tagListCall: vi.fn(),
  tagCreateCall: vi.fn(),
  tagDeleteCall: vi.fn(),
  modelInfoCall: vi.fn(),
}));

vi.mock("./TagTable", () => {
  const prodTag: Tag = { name: "prod-tag", description: "", models: [], created_at: "", updated_at: "" };
  return {
    __esModule: true,
    default: ({
      isLoading,
      onDelete,
      onEdit,
      onSelectTag,
    }: {
      isLoading?: boolean;
      onDelete: (tagName: string) => void;
      onEdit: (tag: Tag) => void;
      onSelectTag: (tagName: string) => void;
    }) => (
      <div data-testid="tag-table">
        {isLoading ? "table-loading" : "table-loaded"}
        <button data-testid="mock-delete-trigger" onClick={() => onDelete("test-tag")}>
          trigger
        </button>
        <button type="button" onClick={() => onSelectTag("prod-tag")}>
          Open prod-tag
        </button>
        <button type="button" onClick={() => onEdit(prodTag)}>
          Edit prod-tag
        </button>
      </div>
    ),
  };
});

vi.mock("./tag_info", () => ({
  __esModule: true,
  default: ({ tagId, onClose }: { tagId: string; onClose: () => void }) => (
    <div>
      <div data-testid="tag-info">Mock Tag Info View {tagId}</div>
      <button type="button" onClick={onClose}>
        Close tag info
      </button>
    </div>
  ),
}));

vi.mock("./components/CreateTagModal", () => ({
  __esModule: true,
  default: () => <div>Mock Create Tag Modal</div>,
}));

const mockTagListCall = vi.mocked(tagListCall);
const mockTagDeleteCall = vi.mocked(tagDeleteCall);

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

describe("TagManagement loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    renderWithProviders(<TagManagement accessToken={null} userID={null} userRole={null} />);
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockTagListCall).not.toHaveBeenCalled();
  });

  it("should show the loading state until the tag fetch settles", async () => {
    let resolveFetch: (value: Record<string, never>) => void = () => {};
    mockTagListCall.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    renderWithProviders(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({});
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockTagListCall).toHaveBeenCalledWith("sk-test");
  });
});

describe("TagManagement delete flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagListCall.mockResolvedValue({});
  });

  it("should confirm deletion through the shared DeleteResourceModal and call tagDeleteCall with the tag name", async () => {
    const user = userEvent.setup();
    mockTagDeleteCall.mockResolvedValue(undefined);
    renderWithProviders(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await screen.findByText("table-loaded");

    expect(screen.queryByText("Tag Information")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("mock-delete-trigger"));

    expect(await screen.findByText("Tag Information")).toBeInTheDocument();
    expect(screen.getByText("test-tag")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /delete/i }));

    expect(mockTagDeleteCall).toHaveBeenCalledWith("sk-test", "test-tag");
  });

  it("should not call tagDeleteCall when the deletion is cancelled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await screen.findByText("table-loaded");

    await user.click(screen.getByTestId("mock-delete-trigger"));
    await screen.findByText("Tag Information");

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(mockTagDeleteCall).not.toHaveBeenCalled();
  });
});

describe("TagManagement ?tag= detail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagListCall.mockResolvedValue({});
  });

  it("opens the tag named in ?tag= instead of the table", () => {
    renderWithProviders(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />, {
      searchParams: "?tag=prod-tag",
    });

    expect(screen.getByTestId("tag-info")).toHaveTextContent("Mock Tag Info View prod-tag");
    expect(screen.queryByTestId("tag-table")).not.toBeInTheDocument();
  });

  it("pushes ?tag= without ?edit= when a tag name is clicked", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />, {
      searchParams: "?edit=true",
      onUrlUpdate,
    });

    await user.click(await screen.findByRole("button", { name: "Open prod-tag" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tag")).toBe("prod-tag"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("edit")).toBe(false);
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    expect(screen.getByTestId("tag-info")).toHaveTextContent("prod-tag");
  });

  it("pushes ?tag= and ?edit=true together when Edit is chosen in the table", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />, { onUrlUpdate });

    await user.click(await screen.findByRole("button", { name: "Edit prod-tag" }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledTimes(1));
    const update = lastUrlUpdate(onUrlUpdate);
    expect(update?.searchParams.get("tag")).toBe("prod-tag");
    expect(update?.searchParams.get("edit")).toBe("true");
    expect(update?.options.history).toBe("push");
  });

  it("clears ?tag= and ?edit= when the detail view is closed", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />, {
      searchParams: "?tag=prod-tag&edit=true",
      onUrlUpdate,
    });

    await user.click(screen.getByRole("button", { name: "Close tag info" }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const update = lastUrlUpdate(onUrlUpdate);
    expect(update?.searchParams.has("tag")).toBe(false);
    expect(update?.searchParams.has("edit")).toBe(false);
    expect(await screen.findByTestId("tag-table")).toBeInTheDocument();
  });
});
