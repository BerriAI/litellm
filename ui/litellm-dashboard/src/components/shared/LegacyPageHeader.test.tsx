import { renderWithProviders, screen } from "@/../tests/test-utils";
import { describe, expect, it } from "vitest";

import { LegacyPageHeader } from "./LegacyPageHeader";

describe("LegacyPageHeader", () => {
  it("should render the title as a heading", () => {
    renderWithProviders(<LegacyPageHeader title="Virtual Keys" />);

    expect(screen.getByRole("heading", { name: "Virtual Keys" })).toBeInTheDocument();
  });

  it("should render the optional identity and actions", () => {
    renderWithProviders(
      <LegacyPageHeader
        title="Virtual Keys"
        subtitle="Every key that authenticates requests"
        icon={<span>Key icon</span>}
        actions={<button>Create New Key</button>}
      />,
    );

    expect(screen.getByText("Every key that authenticates requests")).toBeInTheDocument();
    expect(screen.getByText("Key icon")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create New Key" })).toBeInTheDocument();
  });

  it("should omit optional actions when none are provided", () => {
    renderWithProviders(<LegacyPageHeader title="Virtual Keys" />);

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
