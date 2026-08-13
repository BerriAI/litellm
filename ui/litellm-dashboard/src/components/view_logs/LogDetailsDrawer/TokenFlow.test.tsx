import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TokenFlow } from "./TokenFlow";

describe("TokenFlow", () => {
  it("should render the total followed by its prompt and completion breakdown", () => {
    render(<TokenFlow prompt={9} completion={3} total={12} />);

    expect(screen.getByText("12 (9 prompt tokens + 3 completion tokens)")).toBeInTheDocument();
  });

  it("should group large counts with thousands separators", () => {
    render(<TokenFlow prompt={1234567} completion={89012} total={1323579} />);

    expect(screen.getByText("1,323,579 (1,234,567 prompt tokens + 89,012 completion tokens)")).toBeInTheDocument();
  });

  it("should fall back to zero for counts the log entry does not carry", () => {
    render(<TokenFlow total={12} />);

    expect(screen.getByText("12 (0 prompt tokens + 0 completion tokens)")).toBeInTheDocument();
  });
});
