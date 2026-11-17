import * as fs from "fs";
import { ADMIN_STORAGE_PATH } from "../constants";

/**
 * Whether the proxy under test is licensed, read from the admin session JWT's `premium_user`
 * claim. That is the same value the dashboard reads to enable premium-gated controls, so it
 * describes the proxy Playwright is pointed at rather than the environment the runner happens
 * to have, which are not the same machine when E2E_UI_BASE_URL points elsewhere.
 */
export function proxyIsPremium(): boolean {
  const storage = JSON.parse(fs.readFileSync(ADMIN_STORAGE_PATH, "utf-8")) as {
    cookies?: { name: string; value: string }[];
  };
  const token = storage.cookies?.find((cookie) => cookie.name === "token")?.value;
  const payload = token?.split(".")[1];
  if (!payload) {
    return false;
  }
  return JSON.parse(Buffer.from(payload, "base64url").toString("utf-8")).premium_user === true;
}
