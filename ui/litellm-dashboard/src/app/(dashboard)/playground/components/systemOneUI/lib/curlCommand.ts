export interface CurlCommandInput {
  baseUrl: string;
  endpoint: "/v1/decisions" | "/v1/systemone";
  payload: string;
}

export const CURL_KEY_PLACEHOLDER = "$LITELLM_API_KEY";

const shellSingleQuote = (value: string): string => `'${value.replace(/'/g, `'\\''`)}'`;

const prettyPayload = (payload: string): string => {
  try {
    return JSON.stringify(JSON.parse(payload), null, 2);
  } catch {
    return payload;
  }
};

export const buildCurlCommand = ({ baseUrl, endpoint, payload }: CurlCommandInput): string =>
  [
    `curl -X POST ${shellSingleQuote(`${baseUrl.replace(/\/+$/, "")}${endpoint}`)}`,
    `  -H "Authorization: Bearer ${CURL_KEY_PLACEHOLDER}"`,
    `  -H "Content-Type: application/json"`,
    `  -d ${shellSingleQuote(prettyPayload(payload))}`,
  ].join(" \\\n");

export const curlPayload = (command: string): string | undefined => {
  const match = /-d '([\s\S]*)'$/.exec(command);
  return match === null ? undefined : match[1].replace(/'\\''/g, "'");
};
