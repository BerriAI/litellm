import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { tagDeleteCall, tagListCall } from "@/components/networking";

import TagManagement from "./index";

vi.mock("@/components/networking", () => ({
  tagListCall: vi.fn(),
  tagCreateCall: vi.fn(),
  tagDeleteCall: vi.fn(),
  modelInfoCall: vi.fn(),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

vi.mock("./TagTable", () => ({
  __esModule: true,
  default: ({ isLoading, onDelete }: { isLoading?: boolean; onDelete: (tagName: string) => void }) => (
    <div data-testid="tag-table">
      {isLoading ? "table-loading" : "table-loaded"}
      <button data-testid="mock-delete-trigger" onClick={() => onDelete("test-tag")}>
        trigger
      </button>
    </div>
  ),
}));

vi.mock("./tag_info", () => ({
  __esModule: true,
  default: () => <div>Mock Tag Info View</div>,
}));

vi.mock("./components/CreateTagModal", () => ({
  __esModule: true,
  default: () => <div>Mock Create Tag Modal</div>,
}));

const mockTagListCall = vi.mocked(tagListCall);
const mockTagDeleteCall = vi.mocked(tagDeleteCall);

describe("TagManagement loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    render(<TagManagement accessToken={null} userID={null} userRole={null} />);
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
    render(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
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
    mockTagDeleteCall.mockResolvedValue({});
    render(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
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
    render(<TagManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await screen.findByText("table-loaded");

    await user.click(screen.getByTestId("mock-delete-trigger"));
    await screen.findByText("Tag Information");

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(mockTagDeleteCall).not.toHaveBeenCalled();
  });
});
