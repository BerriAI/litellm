import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import SSOSettingsLoadingSkeleton from "./SSOSettingsLoadingSkeleton";

// Shadcn Skeleton is a <div className="animate-pulse rounded-md bg-muted">.
// Shadcn Card is a <div> with a "rounded-lg border bg-card" class.
// The lucide Shield icon renders an <svg class="lucide lucide-shield">.

describe("SSOSettingsLoadingSkeleton", () => {
  it("should render without crashing", () => {
    expect(() => render(<SSOSettingsLoadingSkeleton />)).not.toThrow();
  });

  it("should render Card component", () => {
    const { container } = render(<SSOSettingsLoadingSkeleton />);
    // Card has the "rounded-lg border" Tailwind class (see @/components/ui/card).
    expect(container.querySelector(".rounded-lg.border")).toBeInTheDocument();
  });

  it("should render Space component with correct props", () => {
    const { container } = render(<SSOSettingsLoadingSkeleton />);
    // The "flex flex-col gap-6" wrapper inside Card replaces the antd Space
    // vertical layout.
    expect(container.querySelector(".flex.flex-col.gap-6")).toBeInTheDocument();
  });

  describe("Header Section", () => {
    it("should render Shield icon", () => {
      const { container } = render(<SSOSettingsLoadingSkeleton />);
      const shield = container.querySelector(".lucide-shield");
      expect(shield).toBeInTheDocument();
      expect(shield).toHaveClass("w-6", "h-6", "text-muted-foreground");
    });

    it("should render title with correct text and level", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const title = screen.getByRole("heading", { level: 3, name: "SSO Configuration" });
      expect(title).toBeInTheDocument();
    });

    it("should render subtitle text", () => {
      render(<SSOSettingsLoadingSkeleton />);
      expect(
        screen.getByText("Manage Single Sign-On authentication settings"),
      ).toBeInTheDocument();
    });

    it("should render two skeleton buttons with correct styles", () => {
      const { container } = render(<SSOSettingsLoadingSkeleton />);
      // The two header skeletons are in a flex container next to the
      // heading. They use Tailwind width classes.
      const headerSkeletons = container.querySelectorAll(".animate-pulse.h-8");
      expect(headerSkeletons.length).toBe(2);

      const widths = Array.from(headerSkeletons).map((el) => el.className);
      expect(widths.some((c) => c.includes("w-[170px]"))).toBe(true);
      expect(widths.some((c) => c.includes("w-[190px]"))).toBe(true);
    });
  });

  describe("Descriptions Table", () => {
    it("should render Descriptions component with bordered prop", () => {
      const { container } = render(<SSOSettingsLoadingSkeleton />);
      // The descriptions-table replacement has an outer bordered container.
      const tableWrapper = container.querySelector(".border.border-border.rounded-md");
      expect(tableWrapper).toBeInTheDocument();
    });

    it("should apply correct column configuration", () => {
      const { container } = render(<SSOSettingsLoadingSkeleton />);
      // Each row uses a two-column grid with the label column capped at 200px.
      const rows = container.querySelectorAll(
        ".grid.grid-cols-\\[minmax\\(120px\\,200px\\)_1fr\\]",
      );
      expect(rows.length).toBe(5);
    });

    it("should render exactly 5 description items", () => {
      const { container } = render(<SSOSettingsLoadingSkeleton />);
      const rows = container.querySelectorAll(
        ".grid.grid-cols-\\[minmax\\(120px\\,200px\\)_1fr\\]",
      );
      expect(rows.length).toBe(5);
    });

    describe("Description Items Structure", () => {
      it("should render exactly 10 skeleton nodes total", () => {
        const { container } = render(<SSOSettingsLoadingSkeleton />);
        // 5 label skeletons + 5 content skeletons = 10. Exclude the 2 header
        // skeletons which have `h-8`.
        const allSkeletons = container.querySelectorAll(".animate-pulse.h-4");
        expect(allSkeletons.length).toBe(10);
      });

      it("should render 5 skeleton nodes for labels with width 80", () => {
        const { container } = render(<SSOSettingsLoadingSkeleton />);
        // Label skeletons use the `w-20` Tailwind class (5rem = 80px).
        const labelSkeletons = container.querySelectorAll(".animate-pulse.h-4.w-20");
        expect(labelSkeletons.length).toBe(5);
      });

      it("should render skeleton nodes for content with correct widths", () => {
        const { container } = render(<SSOSettingsLoadingSkeleton />);
        // Content skeletons have a pixel width passed via inline style.
        const expectedWidths = ["100px", "200px", "250px", "180px", "220px"];
        const contentSkeletons = Array.from(
          container.querySelectorAll<HTMLElement>(".animate-pulse.h-4"),
        ).filter((el) => el.style.width !== "");
        const widths = contentSkeletons.map((el) => el.style.width).sort();
        expect(widths).toEqual(expectedWidths.sort());
      });
    });
  });

  describe("Accessibility and Structure", () => {
    it("should have proper semantic structure", () => {
      const { container } = render(<SSOSettingsLoadingSkeleton />);
      // Card > flex container > description table
      const card = container.querySelector(".rounded-lg.border")!;
      const flex = card.querySelector(".flex.flex-col.gap-6")!;
      const table = container.querySelector(".border.border-border.rounded-md")!;
      expect(card).toBeInTheDocument();
      expect(flex).toBeInTheDocument();
      expect(table).toBeInTheDocument();
      expect(flex.contains(table)).toBe(true);
    });

    it("should render all skeleton elements as active", () => {
      const { container } = render(<SSOSettingsLoadingSkeleton />);
      // Shadcn Skeleton always has the "animate-pulse" class (== active).
      const skeletons = container.querySelectorAll(".animate-pulse");
      expect(skeletons.length).toBeGreaterThan(0);
      skeletons.forEach((s) => {
        expect(s).toHaveClass("animate-pulse");
      });
    });
  });
});
