/**
 * Utility functions for managing cookies
 */

/**
 * Clears the token cookie from both root and /ui paths
 */
export function clearTokenCookies() {
  // Get the current domain
  const domain = window.location.hostname;
  
  // Clear with various combinations of path and SameSite
  const paths = ['/', '/ui'];
  const sameSiteValues = ['Lax', 'Strict', 'None'];
  
  paths.forEach(path => {
    // Basic clearing
    document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path};`;
    
    // With domain
    document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain};`;
    
    // Try different SameSite values
    sameSiteValues.forEach(sameSite => {
      const secureFlag = sameSite === 'None' ? ' Secure;' : '';
      document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; SameSite=${sameSite};${secureFlag}`;
      document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain}; SameSite=${sameSite};${secureFlag}`;
    });
  });
  
  console.log("After clearing cookies:", document.cookie);
}

export function getAuthToken() {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return null;
    }
    
    const tokenPattern = /^token_(\d+)$/;
    const allCookies = document.cookie.split("; ");
    
    const tokenCookies = allCookies
      .map(cookie => {
        const parts = cookie.split("=");
        const name = parts[0];
        const match = name.match(tokenPattern);
        if (match) {
          return {
            name,
            timestamp: parseInt(match[1], 10),
            value: parts.slice(1).join("=")
          };
        }
        return null;
      })
      .filter(cookie => cookie !== null);
    
    if (tokenCookies.length > 0) {
      // Sort by timestamp (newest first)
      tokenCookies.sort((a, b) => b.timestamp - a.timestamp);
      return tokenCookies[0].value;
    }
    
    return null;
}