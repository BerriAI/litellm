import { resolveApiBase } from "./resolveApiBase";

/**
 * Runtime request config the typed client reads on every call. The values are
 * mutable at runtime (base URL can switch to a worker origin; the auth header
 * name and token come from the logged-in session), and they are owned outside
 * this module: networking.tsx registers the base URL / header-name / token
 * getters and the error handler. The token getter reads the session cookie, the
 * same source useAuthorized decodes, so the client's token and the gate that
 * enables a query cannot diverge. Keeping the seam here (not importing from the
 * component tree) lets api.ts stay in lib/http without a layering inversion.
 *
 * The base URL default resolves from NEXT_PUBLIC_BASE_URL so a request still
 * hits the right origin if it fires before networking registers its fuller
 * getter (which additionally folds in the server root path from the live UI
 * config). The auth header name has no build-time source, so it defaults to
 * "Authorization" until the session's JWT supplies a custom one.
 */

type Getter<T> = () => T;

let baseUrlGetter: Getter<string> = () => resolveApiBase({ explicitBase: process.env.NEXT_PUBLIC_BASE_URL });
let authHeaderNameGetter: Getter<string> = () => "Authorization";
let authTokenGetter: Getter<string | null> = () => null;
let errorHandler: (message: string) => void = () => {};

export const registerBaseUrlGetter = (getter: Getter<string>): void => {
  baseUrlGetter = getter;
};

export const registerAuthHeaderNameGetter = (getter: Getter<string>): void => {
  authHeaderNameGetter = getter;
};

export const registerAuthTokenGetter = (getter: Getter<string | null>): void => {
  authTokenGetter = getter;
};

export const registerErrorHandler = (handler: (message: string) => void): void => {
  errorHandler = handler;
};

export const getRequestBaseUrl = (): string => baseUrlGetter();
export const getAuthHeaderName = (): string => authHeaderNameGetter();
export const getAuthToken = (): string | null => authTokenGetter();
export const reportError = (message: string): void => errorHandler(message);
