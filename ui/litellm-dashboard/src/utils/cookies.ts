export function getCookie(name: string, userId?: string): string | null {
    const cookieName = userId ? `${name}_${userId}` : name;
    const cookieValue = document.cookie
      .split("; ")
      .find((row) => row.startsWith(cookieName + "="));
    return cookieValue ? cookieValue.split("=")[1] : null;
  }
  
  export function setCookie(name: string, value: string, userId?: string): void {
    const cookieName = userId ? `${name}_${userId}` : name;
    document.cookie = `${cookieName}=${value}; path=/`;
    if (name === 'token' && userId) {
      cleanupOldTokens(userId);
    }
  }
  
  export function removeCookie(name: string, userId?: string): void {
    const cookieName = userId ? `${name}_${userId}` : name;
    document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
  }
  
  function cleanupOldTokens(currentUserId: string): void {
    const cookies = document.cookie.split(';');
    cookies.forEach(cookie => {
      const [name] = cookie.split('=').map(part => part.trim());
      if (name.startsWith('token_')) {
        // Only keep the token for current userID
        if (!name.includes(currentUserId)) {
          removeCookie(name);
        }
      }
    });
  }