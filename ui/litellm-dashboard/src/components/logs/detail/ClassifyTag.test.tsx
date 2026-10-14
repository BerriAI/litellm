import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ClassifyTag } from "./ClassifyTag";

describe("ClassifyTag", () => {
  it("renders for an auto-router classifier row", () => {
    render(<ClassifyTag origin="autorouter_classifier" />);
    expect(screen.getByText("Classify")).toBeInTheDocument();
  });

  it("renders nothing for ordinary traffic, which is what makes the tag meaningful", () => {
    const { container } = render(<ClassifyTag origin={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for an unrecognized origin rather than labelling it as a classifier call", () => {
    const { container } = render(<ClassifyTag origin="something_else" />);
    expect(container).toBeEmptyDOMElement();
  });
});
