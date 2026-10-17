import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { tagListCall } from "../networking";
import TagSelector from "./TagSelector";

vi.mock("../networking", () => ({
  tagListCall: vi.fn(),
}));

const mockTagListCall = vi.mocked(tagListCall);

describe("TagSelector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagListCall.mockResolvedValue({});
  });

  it("should render the tag selector", () => {
    render(<TagSelector onChange={() => {}} accessToken="test-token" />);
  });

  it("should allow creating new tags", () => {
    const { container } = render(<TagSelector onChange={() => {}} accessToken="test-token" />);
    const tagSelector = container.querySelector("input");
    expect(tagSelector).toBeInTheDocument();
    if (tagSelector) {
      fireEvent.change(tagSelector, { target: { value: "new-tag" } });
      expect(tagSelector).toHaveValue("new-tag");
      fireEvent.keyDown(tagSelector, { key: "Enter" });
      expect(tagSelector).toHaveValue("new-tag");
    }
  });

  it("does not match shared dynamic tag descriptions", async () => {
    mockTagListCall.mockResolvedValue({
      first: {
        name: "dynamic-tag-one",
        description: "This is just a spend tag that was passed dynamically",
        models: [],
        created_at: "2025-01-01",
        updated_at: "2025-01-01",
      },
      second: {
        name: "dynamic-tag-two",
        description: "This is just a spend tag that was passed dynamically",
        models: [],
        created_at: "2025-01-01",
        updated_at: "2025-01-01",
      },
    });
    const user = userEvent.setup();
    render(<TagSelector onChange={() => {}} accessToken="test-token" />);

    await waitFor(() => expect(mockTagListCall).toHaveBeenCalledWith("test-token"));
    const input = screen.getByRole("combobox");
    await waitFor(() => expect(input).toBeEnabled());
    await user.click(input);
    expect(await screen.findByRole("option", { name: /dynamic-tag-one/ })).toBeInTheDocument();
    await user.type(input, "spend tag");

    expect(await screen.findByRole("option", { name: 'Create "spend tag"' })).toBeInTheDocument();
    expect(screen.getAllByRole("option")).toHaveLength(1);
  });
});
