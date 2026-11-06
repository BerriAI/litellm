import React, { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import {
  CustomerRateLimitEditor,
  CustomerRateLimitEntry,
  customerLimitsToRows,
  customerRowsToLimits,
} from "./CustomerRateLimitEditor";

function Harness({
  initial = [] as CustomerRateLimitEntry[],
  onValue,
}: {
  initial?: CustomerRateLimitEntry[];
  onValue?: (rows: CustomerRateLimitEntry[]) => void;
}) {
  const [rows, setRows] = useState<CustomerRateLimitEntry[]>(initial);
  return (
    <CustomerRateLimitEditor
      value={rows}
      onChange={(next) => {
        setRows(next);
        onValue?.(next);
      }}
    />
  );
}

const row = (customer: string, rpm: number | null, tpm: number | null): CustomerRateLimitEntry => ({
  id: `r-${customer}`,
  customer,
  rpm_limit: rpm,
  tpm_limit: tpm,
});

describe("CustomerRateLimitEditor", () => {
  it("renders a customer, RPM and TPM field per row", () => {
    render(<Harness initial={[row("acme", 100, 5000)]} />);
    expect(screen.getByRole("textbox", { name: "Customer" })).toHaveValue("acme");
    expect(screen.getByRole("spinbutton", { name: "RPM limit" })).toHaveValue(100);
    expect(screen.getByRole("spinbutton", { name: "TPM limit" })).toHaveValue(5000);
  });

  it("adds a row when Add Customer Limit is clicked", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("button", { name: /add customer limit/i }));
    expect(screen.getAllByRole("textbox", { name: "Customer" })).toHaveLength(1);
  });

  it("records TPM and RPM as numbers and clears them to null", async () => {
    const user = userEvent.setup();
    const seen: CustomerRateLimitEntry[][] = [];
    render(<Harness initial={[row("acme", 60, null)]} onValue={(v) => seen.push(v)} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "TPM limit" }), { target: { value: "900" } });
    expect(seen[seen.length - 1][0].tpm_limit).toBe(900);

    await user.clear(screen.getByRole("spinbutton", { name: "RPM limit" }));
    expect(seen[seen.length - 1][0].rpm_limit).toBeNull();
  });

  it("removes only the clicked row", async () => {
    const user = userEvent.setup();
    render(<Harness initial={[row("keep", 10, null), row("drop", 20, null)]} />);
    await user.click(screen.getAllByRole("button", { name: "Remove customer limit" })[1]);
    const customers = screen.getAllByRole("textbox", { name: "Customer" });
    expect(customers).toHaveLength(1);
    expect(customers[0]).toHaveValue("keep");
  });
});

describe("customerRowsToLimits", () => {
  it("splits rows into rpm and tpm maps, skipping blank customers and empty cells", () => {
    expect(
      customerRowsToLimits([
        row(" acme ", 10, 5000),
        row("globex", null, 100),
        row("   ", 5, 5),
        row("initech", null, null),
      ]),
    ).toStrictEqual({
      customer_rpm_limit: { acme: 10 },
      customer_tpm_limit: { acme: 5000, globex: 100 },
    });
  });
});

describe("customerLimitsToRows", () => {
  it("merges rpm and tpm maps into one row per customer and drops non-numeric entries", () => {
    const rows = customerLimitsToRows({ acme: 10, bad: "x" }, { acme: 5000, globex: 100 });
    expect(rows.map(({ customer, rpm_limit, tpm_limit }) => ({ customer, rpm_limit, tpm_limit }))).toStrictEqual([
      { customer: "acme", rpm_limit: 10, tpm_limit: 5000 },
      { customer: "globex", rpm_limit: null, tpm_limit: 100 },
    ]);
  });

  it("returns no rows for missing metadata", () => {
    expect(customerLimitsToRows(undefined, null)).toStrictEqual([]);
  });
});
