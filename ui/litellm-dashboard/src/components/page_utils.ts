/**
 * Utility functions for working with navigation pages
 */

import { menuGroups } from "./leftnav";
import { pageDescriptions, PageMetadata } from "./page_metadata";

/**
 * Get all available pages from the navigation menu configuration
 * Used by UI Settings to display available pages for visibility control
 */
export const getAvailablePages = (): PageMetadata[] => {
  const pages: PageMetadata[] = [];

  menuGroups.forEach((group) => {
    group.items.forEach((item) => {
      // Add top-level items (skip parent containers like 'tools', 'experimental', 'settings')
      if (item.page && item.page !== "tools" && item.page !== "experimental" && item.page !== "settings") {
        const label = typeof item.label === "string" ? item.label : item.key;
        pages.push({
          page: item.page,
          label: label,
          group: group.groupLabel,
          description: pageDescriptions[item.page] || "No description available",
        });
      }

      // Add children items
      if (item.children) {
        const parentLabel = typeof item.label === "string" ? item.label : item.key;
        item.children.forEach((child) => {
          const childLabel = typeof child.label === "string" ? child.label : child.key;
          pages.push({
            page: child.page,
            label: childLabel,
            group: `${group.groupLabel} > ${parentLabel}`,
            description: pageDescriptions[child.page] || "No description available",
          });
        });
      }
    });
  });

  return pages;
};
