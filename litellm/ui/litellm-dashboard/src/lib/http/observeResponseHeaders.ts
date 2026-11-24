export type ResponseHeadersListener = (headers: Headers) => void;

export const observeResponseHeaders =
  (onHeaders: ResponseHeadersListener): typeof fetch =>
  async (input, init) => {
    const response = await fetch(input, init);
    onHeaders(response.headers);
    return response;
  };
