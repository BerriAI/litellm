 /**
 * Utility functions for cookie management
 */

/**
 * Get a cookie value by name
 * @param name The name of the cookie to retrieve
 * @returns The cookie value or null if not found
 */
export function getCookie(name: string): string | null {
    const cookieValue = document.cookie
      .split('; ')
      .find(row => row.startsWith(name + '='));
    return cookieValue ? cookieValue.split('=')[1] : null;
  }
  
  /**
   * Set a token cookie, removing any existing ones first
   * @param value The value to set for the token
   * @param options Additional cookie options
   */
  export function setTokenCookie(value: string): void {
    // Delete existing token cookie by setting expiration in the past
    document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    
    // Set new token cookie with proper attributes
    document.cookie = `token=${value}; path=/; max-age=2592000; SameSite=Strict`;
  }