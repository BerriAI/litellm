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

  it("omits the optional slots when not provided", () => {
    render(<PageHeader title="Virtual Keys" />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(document.querySelector("p")).toBeNull();
  });
});
