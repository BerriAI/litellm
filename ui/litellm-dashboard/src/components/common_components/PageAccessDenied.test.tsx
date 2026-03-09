import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PageAccessDenied from "./PageAccessDenied";

describe("PageAccessDenied", () => {
  it("should render restriction and resolution guidance", () => {
    render(<PageAccessDenied pageName="Teams" />);

    expect(screen.getByText("Access to Teams is restricted")).toBeInTheDocument();
    expect(
      screen.getByText(/If you need access to this page, please contact your proxy administrator\./),
    ).toBeInTheDocument();
  });

  it("should not render a navigation CTA", () => {
    render(<PageAccessDenied pageName="Teams" />);
    expect(screen.queryByRole("button", { name: "Go to Virtual Keys" })).not.toBeInTheDocument();
  });
});
