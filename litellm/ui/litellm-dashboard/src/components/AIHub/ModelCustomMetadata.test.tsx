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
    const terms = screen.getAllByRole("term").map((term) => term.textContent);
    const definitions = screen.getAllByRole("definition").map((definition) => definition.textContent);
    expect(terms.map((term, index) => [term, definitions[index]])).toEqual([
      ["retirement_date", "2027-03-01"],
      ["provisioned_ptu", "100"],
      ["gsu", "false"],
    ]);
  });

  it("renders nothing when the model has no tags or metadata", () => {
    const { container } = render(<ModelCustomMetadata customTags={null} customMetadata={{}} />);
    expect(container).toBeEmptyDOMElement();
  });
});
