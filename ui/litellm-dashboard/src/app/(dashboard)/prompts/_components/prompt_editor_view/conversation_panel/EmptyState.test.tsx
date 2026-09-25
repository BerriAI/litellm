import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import EmptyState from "./EmptyState";

describe("EmptyState", () => {
  it("explains when variables must be filled", () => {
    render(<EmptyState hasVariables />);
    expect(screen.getByText(/fill in the variables/i)).toBeInTheDocument();
  });
});
