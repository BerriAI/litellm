/**
 * Utility functions for managing cookies
 */

/**
 * Clears the token cookie from both root and /ui paths
 */
export function clearTokenCookies() {
  // Clear token cookie on root path
  document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
  
  // Clear token cookie on /ui path
  document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/ui;";
  
  console.log("After clearing cookies:", document.cookie);
}

/**
 * Gets a cookie value by name
 * @param name The name of the cookie to retrieve
 * @returns The cookie value or null if not found
 */
export function getCookie(name: string) {
  const cookieValue = document.cookie
    .split('; ')
    .find(row => row.startsWith(name + '='));
  return cookieValue ? cookieValue.split('=')[1] : null;
} 