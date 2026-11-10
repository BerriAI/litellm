import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SSOSettingsEmptyPlaceholder from "./SSOSettingsEmptyPlaceholder";

describe("SSOSettingsEmptyPlaceholder", () => {
  it("should render", () => {
    const onAdd = vi.fn();

    render(<SSOSettingsEmptyPlaceholder onAdd={onAdd} />);

    expect(screen.getByText("No SSO Configuration Found")).toBeInTheDocument();
    expect(screen.getByText("Configure SSO")).toBeInTheDocument();
  });
});
