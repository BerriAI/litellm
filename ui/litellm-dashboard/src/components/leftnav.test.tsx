import { act, fireEvent, render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import Sidebar from "./leftnav";

// Stub ResizeObserver used by antd in jsdom
class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(global as any).ResizeObserver = ResizeObserver;

vi.mock("../utils/roles", () => {
  return {
    all_admin_roles: ["admin"],
    internalUserRoles: ["internal"],
    rolesWithWriteAccess: ["admin", "internal"],
    isAdminRole: (role: string) => role === "admin",
  };
});

describe("Sidebar (leftnav)", () => {
  const defaultProps = {
    accessToken: null as string | null,
    setPage: vi.fn(),
    userRole: "admin",
    defaultSelectedKey: "api-keys",
    collapsed: false,
  };

  it("renders all top-level (non-nested) tabs for admin", () => {
    const { getByText } = render(<Sidebar {...defaultProps} />);

    const topLevelLabels = [
      "Virtual Keys",
      "Playground",
      "Models + Endpoints",
      "Usage",
      "Teams",
      "Organizations",
      "Internal Users",
      "Budgets",
      "API Reference",
      "AI Hub",
      "Logs",
      "Guardrails",
      "MCP Servers",
      "Tools",
      "Experimental",
      "Settings",
    ];

    topLevelLabels.forEach((label) => {
      expect(getByText(label)).toBeInTheDocument();
    });
  });

  it("expands a nested tab to reveal its children (Tools > Search Tools)", async () => {
    const { getByText, queryByText } = render(<Sidebar {...defaultProps} />);

    expect(queryByText("Search Tools")).not.toBeInTheDocument();
    act(() => {
      fireEvent.click(getByText("Tools"));
    });
    await waitFor(() => {
      expect(getByText("Search Tools")).toBeInTheDocument();
    });
  });
  it("has no duplicate keys among all menu items and their children", () => {
    // Helper to recursively extract all keys from Ant Design Menu items
    function getAllKeysFromMenu(wrapper: HTMLElement): string[] {
      const allKeys: string[] = [];
      // Ant Design renders key as data-menu-id or inside attributes, but for this case, we look for text as fallback.
      // For a generic check, here we fetch ids from rendered list items, and also descend into submenus
      const items = wrapper.querySelectorAll("[data-menu-id]");
      items.forEach((item) => {
        const dataMenuId = item.getAttribute("data-menu-id");
        if (dataMenuId) {
          allKeys.push(dataMenuId);
        }
      });
      return allKeys;
    }

    const { container } = render(<Sidebar {...defaultProps} />);
    const allRenderedKeys = getAllKeysFromMenu(container);

    const keySet = new Set<string>();
    const duplicates: string[] = [];
    for (const key of allRenderedKeys) {
      if (keySet.has(key)) {
        duplicates.push(key);
      }
      keySet.add(key);
    }
    expect(duplicates).toHaveLength(0);
  });
});
