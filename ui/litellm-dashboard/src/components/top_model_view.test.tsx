import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TopModelView from "./top_model_view";

describe("TopModelView", () => {
  it("should render", () => {
    const { container } = render(<TopModelView topModels={[]} />);
    expect(container).toBeTruthy();
  });

  it("should have a table view button", () => {
    const { getByText } = render(<TopModelView topModels={[]} />);
    expect(getByText("Table View")).toBeInTheDocument();
  });

  it("should have a chart view", () => {
    const { getByText } = render(<TopModelView topModels={[]} />);
    expect(getByText("Chart View")).toBeInTheDocument();
  });

  ["Model", "Spend (USD)", "Successful", "Failed", "Tokens"].forEach((header) => {
    it(`should have a ${header} column`, () => {
      const { getByText } = render(<TopModelView topModels={[]} />);
      expect(getByText(header)).toBeInTheDocument();
    });
  });

  it("should show the model's information on the table", () => {
    const { getByText } = render(
      <TopModelView
        topModels={[
          {
            key: "gpt-4",
            spend: 150.5,
            successful_requests: 100,
            failed_requests: 5,
            tokens: 50000,
          },
        ]}
      />,
    );
    expect(getByText("gpt-4")).toBeInTheDocument();
    expect(getByText("$150.50")).toBeInTheDocument();
    expect(getByText("100")).toBeInTheDocument();
    expect(getByText("5")).toBeInTheDocument();
    expect(getByText("50,000")).toBeInTheDocument();
  });
});
