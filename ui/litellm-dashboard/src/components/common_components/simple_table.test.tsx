import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SimpleTable, type SimpleTableColumn } from "./simple_table";

interface Row {
  name: string;
  spend: number;
}

const columns: SimpleTableColumn<Row>[] = [
  { header: "Name", accessor: "name" },
  { header: "Spend", accessor: "spend", numeric: true },
];

describe("SimpleTable numeric columns", () => {
  it("right-aligns the header and cells of a numeric column only", () => {
    render(<SimpleTable data={[{ name: "Alice", spend: 42 }]} columns={columns} />);

    expect(screen.getByRole("columnheader", { name: "Spend" })).toHaveClass("text-right", "tabular-nums");
    expect(screen.getByText("42").closest("td")).toHaveClass("text-right", "tabular-nums");
    expect(screen.getByRole("columnheader", { name: "Name" })).not.toHaveClass("text-right");
    expect(screen.getByText("Alice").closest("td")).not.toHaveClass("text-right");
  });
});
