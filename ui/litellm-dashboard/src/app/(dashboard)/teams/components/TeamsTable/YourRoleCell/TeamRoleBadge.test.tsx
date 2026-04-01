import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TeamRoleBadge from "./TeamRoleBadge";

const renderBadge = (role: string | null) => render(<div data-testid="wrap">{TeamRoleBadge(role)}</div>);

describe("TeamRoleBadge", () => {
  it("renders admin badge with correct label, base classes, styles, and an icon", () => {
    renderBadge("admin");
    const label = screen.getByText("Admin");
    const badge = label.closest("span")!;
    expect(badge).toHaveClass("inline-flex", "items-center", "border", "text-xs", "font-medium");
    expect(badge).toHaveStyle({
      backgroundColor: "#EEF2FF",
      color: "#3730A3",
      borderColor: "#C7D2FE",
    });
    expect(badge.querySelector("svg")).toBeInTheDocument(); // ShieldIcon renders as an SVG
  });

  it.each<[string | null]>([["user"], [null], ["viewer" as unknown as string]])(
    "renders member badge for non-admin role (%p) with correct styles",
    (role) => {
      renderBadge(role);
      const label = screen.getByText("Member");
      const badge = label.closest("span")!;
      expect(badge).toHaveClass("inline-flex", "items-center", "border", "text-xs", "font-medium");
      expect(badge).toHaveStyle({
        backgroundColor: "#F3F4F6",
        color: "#4B5563",
        borderColor: "#E5E7EB",
      });
      expect(badge.querySelector("svg")).toBeInTheDocument(); // UserIcon renders as an SVG
    },
  );
});
