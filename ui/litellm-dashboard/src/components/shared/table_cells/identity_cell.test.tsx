import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { IdentityCell } from "./identity_cell";

describe("IdentityCell", () => {
  it("renders the title", () => {
    render(<IdentityCell title="prod-gateway" />);
    expect(screen.getByText("prod-gateway")).toBeInTheDocument();
  });

  it("renders the subtitle and an inline badge together", () => {
    render(<IdentityCell title="prod-gateway" subtitle="sk-...v0Pw" badge={<span>Active</span>} />);
    expect(screen.getByText("sk-...v0Pw")).toBeInTheDocument();
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("omits the subtitle row when there is no subtitle or badge", () => {
    render(<IdentityCell title="a" />);
    expect(document.querySelector("span.font-mono")).toBeNull();
  });

  it("renders a static div (no button) when not clickable", () => {
    render(<IdentityCell title="a" subtitle="b" />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders a clickable button that signals interactivity and fires onClick", async () => {
    const onClick = vi.fn();
    const user = userEvent.setup();
    render(<IdentityCell title="prod-gateway" subtitle="sk-...v0Pw" onClick={onClick} />);
    const button = screen.getByRole("button");
    expect(button.querySelector(".lucide-chevron-right")).not.toBeNull();
    // The clickable area must read as clickable: a hover background and a pointer cursor.
    expect(button.className).toContain("hover:bg-muted");
    expect(button.className).toContain("cursor-pointer");
    await user.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
