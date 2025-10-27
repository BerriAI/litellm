/**
 * Utility functions for managing cookies
 */

/**
 * Clears the token cookie from both root and /ui paths
 */
export function clearTokenCookies() {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return;
  }

  // Get the current domain
  const domain = window.location.hostname;

  // Clear with various combinations of path and SameSite
  // Include current path in case of custom server root path
  const currentPath = window.location.pathname;
  const paths = ["/", "/ui"];

  // Add the current path directory if it's different from root and /ui
  if (currentPath && currentPath !== "/" && !currentPath.startsWith("/ui")) {
    const dirPath = currentPath.substring(0, currentPath.lastIndexOf("/") + 1);
    if (dirPath && !paths.includes(dirPath)) {
      paths.push(dirPath);
    }
  }

  const sameSiteValues = ["Lax", "Strict", "None"];

  paths.forEach((path) => {
    // Basic clearing
    document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path};`;

    // With domain
    document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain};`;

    // Try different SameSite values
    sameSiteValues.forEach((sameSite) => {
      const secureFlag = sameSite === "None" ? " Secure;" : "";
      document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; SameSite=${sameSite};${secureFlag}`;
      document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain}; SameSite=${sameSite};${secureFlag}`;
    });
  });

  console.log("After clearing cookies:", document.cookie);
}

/**
 * Gets a cookie value by name
 * @param name The name of the cookie to retrieve
 * @returns The cookie value or null if not found
 */
export function getCookie(name: string) {
  if (typeof document === "undefined") return null;
  const cookieValue = document.cookie.split("; ").find((row) => row.startsWith(name + "="));
  return cookieValue ? cookieValue.split("=")[1] : null;
}
