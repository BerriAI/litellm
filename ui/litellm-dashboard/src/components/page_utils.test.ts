/**
 * Tests to ensure page metadata stays in sync with leftnav configuration
 * This catches issues when leftnav structure changes but page_utils/page_metadata aren't updated
 */

import { describe, it, expect } from "vitest";
import { getAvailablePages } from "./page_utils";
import { menuGroups } from "./leftnav";
import { pageDescriptions } from "./page_metadata";
import { internalUserRoles } from "@/utils/roles";

/**
 * Check if a page is accessible to internal users
 * A page is accessible if:
 * 1. It has no role restrictions, OR
 * 2. Its roles include at least one internal user role
 */
const isPageAccessibleToInternalUsers = (pageRoles?: string[]): boolean => {
  if (!pageRoles || pageRoles.length === 0) {
    return true; // No role restrictions
  }
  
  // Check if any of the page's roles match internal user roles
  return pageRoles.some(role => internalUserRoles.includes(role));
};

describe("Page Utils - LeftNav Sync", () => {
  it("should return all pages from leftnav configuration", () => {
    const availablePages = getAvailablePages();
    
    // Should have pages
    expect(availablePages.length).toBeGreaterThan(0);
    
    // Each page should have required fields
    availablePages.forEach((page) => {
      expect(page).toHaveProperty("page");
      expect(page).toHaveProperty("label");
      expect(page).toHaveProperty("group");
      expect(page).toHaveProperty("description");
      expect(typeof page.page).toBe("string");
      expect(typeof page.label).toBe("string");
      expect(typeof page.group).toBe("string");
      expect(typeof page.description).toBe("string");
    });
  });

  it("should include all navigable pages from menuGroups", () => {
    const availablePages = getAvailablePages();
    const availablePageKeys = availablePages.map((p) => p.page);
    
    // Collect all page keys from menuGroups (excluding parent containers and pages not accessible to internal users)
    const menuPageKeys: string[] = [];
    const excludedParents = ["tools", "experimental", "settings"];
    
    menuGroups.forEach((group) => {
      group.items.forEach((item) => {
        if (
          item.page &&
          !excludedParents.includes(item.page) &&
          isPageAccessibleToInternalUsers(item.roles)
        ) {
          menuPageKeys.push(item.page);
        }
        
        // Add children (only if accessible to internal users)
        if (item.children) {
          item.children.forEach((child) => {
            if (isPageAccessibleToInternalUsers(child.roles)) {
              menuPageKeys.push(child.page);
            }
          });
        }
      });
    });
    
    // Every menu page accessible to internal users should be in available pages
    menuPageKeys.forEach((pageKey) => {
      expect(
        availablePageKeys,
        `Page "${pageKey}" from menuGroups should be in getAvailablePages() output`
      ).toContain(pageKey);
    });
  });

  it("should not include parent container pages (tools, experimental, settings)", () => {
    const availablePages = getAvailablePages();
    const availablePageKeys = availablePages.map((p) => p.page);
    
    const excludedParents = ["tools", "experimental", "settings"];
    
    excludedParents.forEach((parent) => {
      expect(
        availablePageKeys,
        `Parent container "${parent}" should not be in available pages`
      ).not.toContain(parent);
    });
  });

  it("should have descriptions for all pages", () => {
    const availablePages = getAvailablePages();
    
    availablePages.forEach((page) => {
      expect(
        page.description,
        `Page "${page.page}" should have a description`
      ).toBeTruthy();
      
      expect(
        page.description,
        `Page "${page.page}" should not have placeholder description`
      ).not.toBe("No description available");
    });
  });

  it("should have pageDescriptions entry for all navigable pages in menuGroups", () => {
    // Collect all page keys from menuGroups
    const menuPageKeys: string[] = [];
    const excludedParents = ["tools", "experimental", "settings"];
    
    menuGroups.forEach((group) => {
      group.items.forEach((item) => {
        if (item.page && !excludedParents.includes(item.page)) {
          menuPageKeys.push(item.page);
        }
        
        if (item.children) {
          item.children.forEach((child) => {
            menuPageKeys.push(child.page);
          });
        }
      });
    });
    
    // Every menu page should have a description
    const missingDescriptions: string[] = [];
    menuPageKeys.forEach((pageKey) => {
      if (!pageDescriptions[pageKey]) {
        missingDescriptions.push(pageKey);
      }
    });
    
    expect(
      missingDescriptions,
      `These pages are missing descriptions in page_metadata.ts: ${missingDescriptions.join(", ")}`
    ).toHaveLength(0);
  });

  it("should not have orphaned descriptions (descriptions for pages not in menuGroups)", () => {
    // Collect all page keys from menuGroups
    const menuPageKeys: string[] = [];
    const excludedParents = ["tools", "experimental", "settings"];
    
    menuGroups.forEach((group) => {
      group.items.forEach((item) => {
        if (item.page && !excludedParents.includes(item.page)) {
          menuPageKeys.push(item.page);
        }
        
        if (item.children) {
          item.children.forEach((child) => {
            menuPageKeys.push(child.page);
          });
        }
      });
    });
    
    // Check for descriptions that don't match any menu page
    const orphanedDescriptions: string[] = [];
    Object.keys(pageDescriptions).forEach((descKey) => {
      if (!menuPageKeys.includes(descKey)) {
        orphanedDescriptions.push(descKey);
      }
    });
    
    expect(
      orphanedDescriptions,
      `These descriptions don't match any page in menuGroups: ${orphanedDescriptions.join(", ")}. Remove them or add the pages to leftnav.`
    ).toHaveLength(0);
  });

  it("should have proper group hierarchy for nested pages", () => {
    const availablePages = getAvailablePages();
    
    // Find pages that should be nested (children of Tools, Experimental, Settings)
    const nestedPages = availablePages.filter((page) => 
      page.group.includes(" > ")
    );
    
    // Each nested page should have parent > child format
    nestedPages.forEach((page) => {
      const parts = page.group.split(" > ");
      expect(
        parts.length,
        `Nested page "${page.page}" should have exactly 2 parts in group hierarchy`
      ).toBe(2);
      
      // Parent should be one of the group labels
      const parentGroup = parts[0];
      const groupLabels = menuGroups.map((g) => g.groupLabel);
      expect(
        groupLabels,
        `Parent group "${parentGroup}" for page "${page.page}" should be a valid group label`
      ).toContain(parentGroup);
    });
  });

  it("should have unique page keys", () => {
    const availablePages = getAvailablePages();
    const pageKeys = availablePages.map((p) => p.page);
    const uniquePageKeys = new Set(pageKeys);
    
    expect(
      pageKeys.length,
      "All page keys should be unique (no duplicates)"
    ).toBe(uniquePageKeys.size);
  });

  it("should match the structure expected by PageVisibilitySettings component", () => {
    const availablePages = getAvailablePages();
    
    // Group pages by their group (same logic as in PageVisibilitySettings)
    const grouped: Record<string, typeof availablePages> = {};
    availablePages.forEach((page) => {
      if (!grouped[page.group]) {
        grouped[page.group] = [];
      }
      grouped[page.group].push(page);
    });
    
    // Should have multiple groups
    expect(Object.keys(grouped).length).toBeGreaterThan(1);
    
    // Each group should have at least one page
    Object.entries(grouped).forEach(([groupName, pages]) => {
      expect(
        pages.length,
        `Group "${groupName}" should have at least one page`
      ).toBeGreaterThan(0);
    });
  });
});
