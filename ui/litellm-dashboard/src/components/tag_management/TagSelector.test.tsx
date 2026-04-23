import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import TagSelector from "./TagSelector";

describe("TagSelector", () => {
  it("should render the tag selector", () => {
    render(<TagSelector onChange={() => {}} accessToken="test-token" />);
  });

  it("should allow creating new tags", () => {
    const onChange = vi.fn();
    const { container } = render(
      <TagSelector onChange={onChange} accessToken="test-token" />,
    );
    const tagSelector = container.querySelector("input");
    expect(tagSelector).toBeInTheDocument();
    if (tagSelector) {
      fireEvent.change(tagSelector, { target: { value: "new-tag" } });
      expect(tagSelector).toHaveValue("new-tag");
      fireEvent.keyDown(tagSelector, { key: "Enter" });
      // Post phase-1: pressing Enter commits the tag (clears the input);
      // onChange fires with the new tag.
      expect(onChange).toHaveBeenCalledWith(["new-tag"]);
    }
  });
});
