const V4 = "(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)){3}";
const V6SEG = "[a-fA-F\\d]{1,4}";
const V6 = `(?:${[
  `(?:${V6SEG}:){7}(?:${V6SEG}|:)`,
  `(?:${V6SEG}:){6}(?:${V4}|:${V6SEG}|:)`,
  `(?:${V6SEG}:){5}(?::${V4}|(?::${V6SEG}){1,2}|:)`,
  `(?:${V6SEG}:){4}(?:(?::${V6SEG}){0,1}:${V4}|(?::${V6SEG}){1,3}|:)`,
  `(?:${V6SEG}:){3}(?:(?::${V6SEG}){0,2}:${V4}|(?::${V6SEG}){1,4}|:)`,
  `(?:${V6SEG}:){2}(?:(?::${V6SEG}){0,3}:${V4}|(?::${V6SEG}){1,5}|:)`,
  `(?:${V6SEG}:){1}(?:(?::${V6SEG}){0,4}:${V4}|(?::${V6SEG}){1,6}|:)`,
  `(?::(?:(?::${V6SEG}){0,5}:${V4}|(?::${V6SEG}){1,7}|:))`,
].join("|")})(?:%[0-9a-zA-Z]{1,})?`;

const PROTOCOL = "(?:(?:[a-z]+:)?//)";
const AUTH = "(?:\\S+(?::\\S*)?@)?";
const HOST = "(?:(?:[a-z\\u00a1-\\uffff0-9][-_]*)*[a-z\\u00a1-\\uffff0-9]+)";
const DOMAIN = "(?:\\.(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)*";
const TLD = "(?:\\.(?:[a-z\\u00a1-\\uffff]{2,}))";
const PORT = "(?::\\d{2,5})?";
const PATH = '(?:[/?#][^\\s"]*)?';

export const URL_REGEX = new RegExp(
  `(?:^(?:${PROTOCOL}|www\\.)${AUTH}(?:localhost|${V4}|${V6}|${HOST}${DOMAIN}${TLD})${PORT}${PATH}$)`,
  "i",
);

export const MAX_URL_LENGTH = 2048;

export const isValidUrl = (value: string): boolean => value.length <= MAX_URL_LENGTH && URL_REGEX.test(value);
