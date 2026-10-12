import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModelCustomMetadata } from "./ModelCustomMetadata";

describe("ModelCustomMetadata", () => {
  it("renders each tag and every metadata entry with its value", () => {
    render(
      <ModelCustomMetadata
        customTags={["deprecated", "ptu"]}
        customMetadata={{ retirement_date: "2027-03-01", provisioned_ptu: 100, gsu: false }}
      />,
    );
    expect(screen.getByText("Tags:")).toBeInTheDocument();
    expect(screen.getByText("deprecated")).toBeInTheDocument();
    expect(screen.getByText("ptu")).toBeInTheDocument();
    expect(screen.getByText("retirement_date").nextElementSibling).toHaveTextContent("2027-03-01");
    expect(screen.getByText("provisioned_ptu").nextElementSibling).toHaveTextContent("100");
    expect(screen.getByText("gsu").nextElementSibling).toHaveTextContent("false");
  });

  it("renders nothing when the model has no tags or metadata", () => {
    const { container } = render(<ModelCustomMetadata customTags={null} customMetadata={{}} />);
    expect(container).toBeEmptyDOMElement();
  });
});
