/**
 * Extracts a user-friendly error message from various error formats
 * @param {any} error - The error object or message
 * @returns {string} - A clean error message
 */
export const parseErrorMessage = (error: any): string => {
  if (!error) return "An unknown error occurred";

  // If error is already a string, return it
  if (typeof error === "string") return error;

  // If error has a message property, check if it's a JSON string
  if (error.message) {
    try {
      // Try to parse the error message as JSON
      const parsedError = JSON.parse(error.message);

      // Handle common nested error structures
      if (parsedError.error && parsedError.error.message) {
        return parsedError.error.message;
      }

      // If parsed successfully but no nested message found, stringify it nicely
      return typeof parsedError === "string" ? parsedError : JSON.stringify(parsedError, null, 2);
    } catch (e) {
      // If parsing fails, just return the original message
      return error.message;
    }
  }

  // If error has a response with data
  if (error.response && error.response.data) {
    if (typeof error.response.data === "string") return error.response.data;
    if (error.response.data.message) return error.response.data.message;
    if (error.response.data.error) {
      return typeof error.response.data.error === "string"
        ? error.response.data.error
        : error.response.data.error.message || JSON.stringify(error.response.data.error);
    }
  }

  // Fallback to stringifying the error
  return String(error);
};
