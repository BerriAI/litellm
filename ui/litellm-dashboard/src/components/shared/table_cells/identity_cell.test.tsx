import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { IdentityCell } from "./identity_cell";

describe("IdentityCell", () => {
  it("renders the title", () => {
    render(<IdentityCell title="prod-gateway" />);
    expect(screen.getByText("prod-gateway")).toBeInTheDocument();
  });

  it("renders the subtitle when provided", () => {
    render(<IdentityCell title="prod-gateway" subtitle="sk-...v0Pw" />);
    expect(screen.getByText("sk-...v0Pw")).toBeInTheDocument();
  });

  it("omits the subtitle when it is empty, null, or undefined", () => {
    const { rerender } = render(<IdentityCell title="a" subtitle="" />);
    expect(screen.queryByText("", { selector: "span.font-mono" })).not.toBeInTheDocument();
    rerender(<IdentityCell title="a" subtitle={null} />);
    expect(document.querySelector("span.font-mono")).toBeNull();
    rerender(<IdentityCell title="a" />);
    expect(document.querySelector("span.font-mono")).toBeNull();
  });
});
