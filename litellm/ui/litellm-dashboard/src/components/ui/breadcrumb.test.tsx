import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Breadcrumb, BreadcrumbItem, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "./breadcrumb";

describe("Breadcrumb", () => {
  it("renders a labelled nav and marks the current page", () => {
    render(
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>Observability</BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage>Logs</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>,
    );
    expect(screen.getByRole("navigation", { name: "breadcrumb" })).toBeInTheDocument();
    const page = screen.getByText("Logs");
    expect(page).toHaveAttribute("aria-current", "page");
    expect(page).toHaveAttribute("data-slot", "breadcrumb-page");
  });

  it("renders a presentational separator", () => {
    const { container } = render(
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>A</BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>B</BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>,
    );
    const sep = container.querySelector('[data-slot="breadcrumb-separator"]');
    expect(sep).toHaveAttribute("aria-hidden", "true");
    expect(sep?.querySelector("svg")).not.toBeNull();
  });
});
