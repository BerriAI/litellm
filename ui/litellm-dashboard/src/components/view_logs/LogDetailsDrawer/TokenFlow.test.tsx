import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TokenFlow } from "./TokenFlow";

const localised = (count: number) => count.toLocaleString();

describe("TokenFlow", () => {
  it("should render the total followed by its prompt and completion breakdown", () => {
    render(<TokenFlow prompt={9} completion={3} total={12} />);

    expect(screen.getByText("12 (9 prompt tokens + 3 completion tokens)")).toBeInTheDocument();
  });

  it("should group large counts the way the reader's locale does", () => {
    render(<TokenFlow prompt={1234567} completion={89012} total={1323579} />);

    expect(
      screen.getByText(
        `${localised(1323579)} (${localised(1234567)} prompt tokens + ${localised(89012)} completion tokens)`,
      ),
    ).toBeInTheDocument();
  });

  it("should fall back to zero for counts the log entry does not carry", () => {
    render(<TokenFlow total={12} />);

    expect(screen.getByText("12 (0 prompt tokens + 0 completion tokens)")).toBeInTheDocument();
  });
});
