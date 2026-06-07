/**
 * Runtime request config the typed client reads on every call. The values are
 * mutable at runtime (base URL can switch to a worker origin; the auth header
 * name and token come from the logged-in session), and they are owned outside
 * this module: networking.tsx registers the base URL / header-name getters and
 * AuthContext publishes the token. Keeping the seam here (not importing from the
 * component tree) lets api.ts stay in lib/http without a layering inversion.
 */

type Getter<T> = () => T;

let baseUrlGetter: Getter<string> = () => "";
let authHeaderNameGetter: Getter<string> = () => "Authorization";
let authToken: string | null = null;

export const registerBaseUrlGetter = (getter: Getter<string>): void => {
  baseUrlGetter = getter;
};

export const registerAuthHeaderNameGetter = (getter: Getter<string>): void => {
  authHeaderNameGetter = getter;
};

export const setAuthToken = (token: string | null): void => {
  authToken = token;
};

export const getRequestBaseUrl = (): string => baseUrlGetter();
export const getAuthHeaderName = (): string => authHeaderNameGetter();
export const getAuthToken = (): string | null => authToken;
