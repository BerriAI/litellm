const NATIVE_ORIGIN = 'https://api.anthropic.com';
const ROUTES = new Set(['/v1/messages', '/v1/messages/count_tokens']);
const PRIVATE_HEADERS = new Set([
  'authorization', 'x-api-key', 'api-key', 'cookie', 'proxy-authorization',
  'host', 'content-length', 'forwarded', 'x-forwarded-for', 'x-real-ip',
]);

function destination(value) {
  if (typeof value !== 'string') return undefined;
  const url = new URL(value);
  const local = url.protocol === 'http:' && url.hostname === '127.0.0.1';
  if (url.protocol !== 'https:' && !local) return undefined;
  if (url.username || url.password || url.search || url.hash) return undefined;
  if (url.origin === NATIVE_ORIGIN) return undefined;
  if (!/^(?:\/[A-Za-z0-9._~-]+)*\/?$/.test(url.pathname)) return undefined;
  return url.origin + url.pathname.replace(/\/+$/, '');
}

export function register(on, options = {}) {
  on('http.fetch', async ($, event, next) => {
    const source = new URL(event.url);
    const core = next.origin.plugin === 'engine' && next.origin.tier === 'core';
    if (source.origin !== NATIVE_ORIGIN && !core) return next(event);
    const path = decodeURIComponent(source.pathname).replace(/\/+$/, '');
    const inference = ROUTES.has(path);
    if (!inference) {
      if (source.origin === NATIVE_ORIGIN &&
          /^\/v1\/(messages\/|complete|chat\/|responses)/.test(path)) {
        return { deny: 'LiteLLM does not support this inference route.' };
      }
      return next(event);
    }
    if (source.origin !== NATIVE_ORIGIN || source.username || source.password || source.hash ||
        event.init?.method?.toUpperCase() !== 'POST') {
      return { deny: 'LiteLLM refused an unexpected inference destination or method.' };
    }
    const gateway = destination(options.gatewayUrl);
    if (!gateway) return { deny: 'LiteLLM gateway configuration is missing or invalid.' };
    const key = options.gatewayKey || await $.env.get('LITELLM_API_KEY');
    if (typeof key !== 'string' || !key || key.startsWith('sk-ant-oat') || /[\s\x00-\x1f\x7f]/.test(key)) {
      return { deny: 'LiteLLM gateway authentication is unavailable. Native login is unchanged.' };
    }
    const headers = Object.fromEntries(Object.entries(event.init.headers ?? {}).filter(([name]) =>
      !PRIVATE_HEADERS.has(name.toLowerCase()) && !name.toLowerCase().startsWith('x-litellm-')));
    const result = await next({
      ...event,
      url: gateway + path + source.search,
      init: {
        ...event.init,
        headers: { ...headers, 'x-litellm-api-key': `Bearer ${key}` },
        auth: undefined,
        socketPath: undefined,
        redirect: 'manual',
      },
    });
    if (result.value?.status === 401) {
      return { deny: 'LiteLLM gateway authentication failed. Update your gateway credential; native Claude login is unchanged.' };
    }
    if (result.value?.status === 403) {
      return { deny: 'LiteLLM gateway denied this request. Check model access, budget and policy; native Claude login is unchanged.' };
    }
    if (result.value?.status >= 300 && result.value.status < 400) {
      return { deny: 'LiteLLM gateway redirects are not permitted.' };
    }
    return result;
  }).catch(async () => ({ deny: 'LiteLLM inference routing failed. Direct inference was not selected.' }));
}
