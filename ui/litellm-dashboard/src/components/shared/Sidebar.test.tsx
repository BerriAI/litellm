import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SidebarMenuButton, sidebarMenuButtonVariants } from "./Sidebar";

const CVA_CONFIG_KEYS = ["base", "variants", "defaultVariants"];

describe("sidebarMenuButtonVariants", () => {
  it("emits its base classes rather than the names of its own config keys", () => {
    const emitted = sidebarMenuButtonVariants({}).split(" ");

    expect(emitted).toContain("rounded-md");
    expect(emitted).toContain("text-sidebar-foreground/70");
    expect(CVA_CONFIG_KEYS.filter((key) => emitted.includes(key))).toEqual([]);
  });

  it("applies the isActive variant on top of the base classes", () => {
    const active = sidebarMenuButtonVariants({ isActive: true }).split(" ");

    expect(active).toContain("bg-sidebar-accent");
    expect(active).toContain("rounded-md");
    expect(sidebarMenuButtonVariants({ isActive: false }).split(" ")).not.toContain("bg-sidebar-accent");
  });
});

describe("SidebarMenuButton", () => {
  it("renders the variant classes onto the button", () => {
    render(<SidebarMenuButton isActive>Keys</SidebarMenuButton>);
    const button = screen.getByRole("button", { name: "Keys" });

    expect(button).toHaveClass("bg-sidebar-accent", "rounded-md");
    for (const key of CVA_CONFIG_KEYS) {
      expect(button).not.toHaveClass(key);
    }
  });
});
