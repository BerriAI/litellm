import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PageHeader } from "./PageHeader";

describe("PageHeader", () => {
  it("renders the title as a heading", () => {
    render(<PageHeader title="Virtual Keys" />);
    expect(screen.getByRole("heading", { name: "Virtual Keys" })).toBeInTheDocument();
  });

  it("renders the subtitle, icon, and actions when provided", () => {
    render(
      <PageHeader
        title="Virtual Keys"
        subtitle="Every key that authenticates requests"
        icon={<svg data-testid="icon" />}
        actions={<button>Create New Key</button>}
      />,
    );
    expect(screen.getByText("Every key that authenticates requests")).toBeInTheDocument();
    expect(screen.getByTestId("icon")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create New Key" })).toBeInTheDocument();
  });

  it("renders leadingActions in the same row as the title, before the trailing actions", () => {
    render(
      <PageHeader
        title="Virtual Keys"
        leadingActions={<button>Create New Key</button>}
        actions={<button>Refresh</button>}
      />,
    );

    const heading = screen.getByRole("heading", { name: "Virtual Keys" });
    const leading = screen.getByRole("button", { name: "Create New Key" });
    const trailing = screen.getByRole("button", { name: "Refresh" });

    const titleGroup = heading.parentElement?.parentElement;
    expect(titleGroup).not.toBeNull();
    expect(titleGroup?.contains(leading)).toBe(true);
    expect(titleGroup?.contains(trailing)).toBe(false);
    expect(leading.compareDocumentPosition(trailing) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("omits the optional slots when not provided", () => {
    render(<PageHeader title="Virtual Keys" />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(document.querySelector("p")).toBeNull();
  });
});
