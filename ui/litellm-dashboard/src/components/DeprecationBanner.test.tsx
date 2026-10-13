import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { DeprecationBanner } from "./DeprecationBanner";

describe("DeprecationBanner", () => {
  it("names the deprecated feature in the heading and the body", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    expect(screen.getByText("Workflows is deprecated")).toBeInTheDocument();
    expect(screen.getByText(/Workflows will be removed in the near future/)).toBeInTheDocument();
  });

  it("says only that removal comes in the near future, with no date", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    expect(screen.getByRole("alert")).toHaveTextContent(
      /^Workflows is deprecatedWorkflows will be removed in the near future\. If you rely on this feature, please share feedback on the deprecation discussion\.$/,
    );
  });

  it("links to the deprecation discussion in a new tab without leaking the opener", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    const link = screen.getByRole("link", { name: "deprecation discussion" });
    expect(link).toHaveAttribute("href", "https://github.com/BerriAI/litellm/discussions/32090");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("cannot be dismissed", async () => {
    const user = userEvent.setup();
    render(<DeprecationBanner featureName="Workflows" />);

    const banner = screen.getByRole("alert");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();

    await user.click(banner);
    await user.click(screen.getByText("Workflows is deprecated"));
    await user.keyboard("{Escape}");

    expect(screen.getByRole("alert")).toHaveTextContent("Workflows is deprecated");
  });
});
