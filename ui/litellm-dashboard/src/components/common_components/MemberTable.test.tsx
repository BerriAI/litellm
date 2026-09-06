import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Member } from "@/components/networking";
import MemberTable, { type MemberTableColumn } from "./MemberTable";

const member: Member = { role: "user", user_id: "u1", user_email: "u1@x.io" };

const extraColumns: MemberTableColumn[] = [
  { title: "Spend (USD)", key: "spend", numeric: true, render: () => <span>$1.50</span> },
  { title: "Joined", key: "joined", render: () => <span>Aug 1</span> },
];

describe("MemberTable numeric columns", () => {
  it("right-aligns the header and cells of a numeric extra column only", () => {
    render(
      <MemberTable
        members={[member]}
        canEdit={false}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        extraColumns={extraColumns}
      />,
    );

    expect(screen.getByText("Spend (USD)").closest("th")).toHaveClass("text-right", "tabular-nums");
    expect(screen.getByText("$1.50").closest("td")).toHaveClass("text-right", "tabular-nums");
    expect(screen.getByText("Joined").closest("th")).not.toHaveClass("text-right");
    expect(screen.getByText("Aug 1").closest("td")).not.toHaveClass("text-right");
  });
});
