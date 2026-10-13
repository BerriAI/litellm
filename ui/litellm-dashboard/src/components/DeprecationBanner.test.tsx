import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DeprecationBanner } from "./DeprecationBanner";

describe("DeprecationBanner", () => {
  it("names the deprecated feature in the heading and the body", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    expect(screen.getByText("Workflows is deprecated")).toBeInTheDocument();
    expect(screen.getByText(/Workflows will be removed in the near future/)).toBeInTheDocument();
  });

  it("does not promise a removal date", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    const banner = screen.getByRole("alert");
    expect(banner).toHaveTextContent("Workflows is deprecated");
    expect(banner).not.toHaveTextContent(/\b\d{4}\b/);
  });

  it("links to the deprecation discussion in a new tab without leaking the opener", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    const link = screen.getByRole("link", { name: "deprecation discussion" });
    expect(link).toHaveAttribute("href", "https://github.com/BerriAI/litellm/discussions/32090");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("cannot be dismissed", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
