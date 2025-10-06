export async function getCountryFromIP(ip: string): Promise<string> {
  try {
    const response = await fetch(`http://ip-api.com/json/${ip}`);
    const data = await response.json();
    console.log("ip lookup data", data);

    // Convert country code to flag emoji (each capital letter is converted to regional indicator symbol)
    const flagEmoji = data.countryCode
      ? data.countryCode
          .toUpperCase()
          .split("")
          .map((char: string) => String.fromCodePoint(char.charCodeAt(0) + 127397))
          .join("")
      : "";

    // Return country name with flag emoji
    return data.country ? `${flagEmoji} ${data.country}` : "Unknown";
  } catch (error) {
    console.error("Error looking up IP:", error);
    return "Unknown";
  }
}
