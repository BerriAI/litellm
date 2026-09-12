import * as oauth from './package/build/index.js';
const origin = 'http://localhost:47078';
const base = process.env.REPRO_BASE_URL || origin;
for (const path of ['/mcp/example', '/example/mcp', '/example', '/mcp', ...(base === origin ? [''] : [])]) {
 const issuer = new URL(base + path);
 const response = await oauth.discoveryRequest(issuer, {
   algorithm: "oauth2",
   [oauth.allowInsecureRequests]: true,
   [oauth.customFetch]: async (url, options) => fetch(String(url).replace(origin, 'http://localhost:4000'), options),
 });
 try {
  const metadata = await oauth.processDiscoveryResponse(issuer, response);
  console.log(JSON.stringify({expected:issuer.href, result:'PASS', issuer:metadata.issuer}));
 } catch (error) {
  console.log(JSON.stringify({expected:issuer.href, result:'REJECT', error:error.message, cause:error.cause}));
 }
}
