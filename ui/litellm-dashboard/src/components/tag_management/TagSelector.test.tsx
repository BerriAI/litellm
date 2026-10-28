import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TagSelector from "./TagSelector";

describe("TagSelector", () => {
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
});
