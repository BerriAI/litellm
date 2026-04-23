/**
 * Utility functions for working with navigation pages
 */

import { menuGroups } from "./leftnav";
import { pageDescriptions, PageMetadata } from "./page_metadata";
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

/**
 * Get all available pages from the navigation menu configuration
 * Used by UI Settings to display available pages for visibility control
 * 
 * IMPORTANT: Only returns pages that internal users can access.
 * Pages restricted to admin-only roles are excluded because internal users
 * cannot see them regardless of the UI visibility setting.
 */
export const getAvailablePages = (): PageMetadata[] => {
  const pages: PageMetadata[] = [];

  menuGroups.forEach((group) => {
    group.items.forEach((item) => {
      // Add top-level items (skip parent containers like 'tools', 'experimental', 'settings')
      // Also skip items that internal users cannot access
      if (
        item.page &&
        item.page !== "tools" &&
        item.page !== "experimental" &&
        item.page !== "settings" &&
        isPageAccessibleToInternalUsers(item.roles)
      ) {
        const label = typeof item.label === "string" ? item.label : item.key;
        pages.push({
          page: item.page,
          label: label,
          group: group.groupLabel,
          description: pageDescriptions[item.page] || "No description available",
        });
      }

      // Add children items (also skip those internal users cannot access)
      if (item.children) {
        const parentLabel = typeof item.label === "string" ? item.label : item.key;
        item.children.forEach((child) => {
          // Include if internal users can access
          if (isPageAccessibleToInternalUsers(child.roles)) {
            const childLabel = typeof child.label === "string" ? child.label : child.key;
            pages.push({
              page: child.page,
              label: childLabel,
              group: `${group.groupLabel} > ${parentLabel}`,
              description: pageDescriptions[child.page] || "No description available",
            });
          }
        });
      }
    });
  });

  return pages;
};
