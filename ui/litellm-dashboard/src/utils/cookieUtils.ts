/**
 * Utility functions for managing cookies
 */

/**
 * Clears the token cookie from both root and /ui paths
 */
import { validateSession } from "../components/networking"

// Define the interface for the JWT token data
export interface JWTTokenData {
  user_id: string;
  user_email: string | null;
  user_role: string;
  login_method: string;
  premium_user: boolean;
  auth_header_name: string;
  iss: string;
  aud: string;
  exp: number;
  disabled_non_admin_personal_key_creation: boolean;
  scopes: string[];
  session_id: string; // ui session id currently in progress
}

export function clearTokenCookies() {
  // Get the current domain
  const domain = window.location.hostname;
  
  // Clear with various combinations of path and SameSite
  const paths = ['/', '/ui'];
  const sameSiteValues = ['Lax', 'Strict', 'None'];
  
  // Get all cookies
  const allCookies = document.cookie.split("; ");
  const tokenPattern = /^token_\d+$/;
  
  // Find all token cookies
  const tokenCookieNames = allCookies
    .map(cookie => cookie.split("=")[0])
    .filter(name => name === "token" || tokenPattern.test(name));
  
  // Clear each token cookie with various combinations
  tokenCookieNames.forEach(cookieName => {
    paths.forEach(path => {
      // Basic clearing
      document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path};`;
      
      // With domain
      document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain};`;
      
      // Try different SameSite values
      sameSiteValues.forEach(sameSite => {
        const secureFlag = sameSite === 'None' ? ' Secure;' : '';
        document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; SameSite=${sameSite};${secureFlag}`;
        document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain}; SameSite=${sameSite};${secureFlag}`;
      });
    });
  });
  
  console.log("After clearing cookies:", document.cookie);
}

export function setAuthToken(token: string) {
  // Generate a token name with current timestamp
  const currentTimestamp = Math.floor(Date.now() / 1000);
  const tokenName = `token_${currentTimestamp}`;
  
  // Set the cookie with the timestamp-based name
  document.cookie = `${tokenName}=${token}; path=/; domain=${window.location.hostname};`;
}

export async function getUISessionDetails(): Promise<JWTTokenData> {
  const validated_jwt_token = await validateSession();
  
  if (validated_jwt_token?.data) {
    return validated_jwt_token.data as JWTTokenData;
  } else {
    throw new Error("Invalid JWT token");
  }
}
