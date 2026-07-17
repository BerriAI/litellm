import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { tagListCall } from "@/components/networking";

import TagManagement from "./index";

vi.mock("@/components/networking", () => ({
  tagListCall: vi.fn(),
  tagCreateCall: vi.fn(),
  tagDeleteCall: vi.fn(),
  modelInfoCall: vi.fn(),
}));

vi.mock("./TagTable", () => ({
  __esModule: true,
  default: ({ isLoading }: { isLoading?: boolean }) => (
    <div data-testid="tag-table">{isLoading ? "table-loading" : "table-loaded"}</div>
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
