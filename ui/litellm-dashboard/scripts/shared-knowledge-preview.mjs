import { createHmac, randomBytes } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer, request as httpRequest } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const dashboardDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const exportDir = resolve(dashboardDir, "out");
const publicDir = resolve(dashboardDir, "public");
const proxyDir = resolve(dashboardDir, "../../litellm/proxy");
const port = Number(process.env.DEMO_PORT ?? 3222);
const previewOrigin = `http://127.0.0.1:${port}`;
const nextOrigin = process.env.DEMO_NEXT_ORIGIN ? new URL(process.env.DEMO_NEXT_ORIGIN) : null;
const sessionKey = `demo-only-${randomBytes(24).toString("hex")}`;
const signingKey = randomBytes(32);
const user = { user_id: "demo-admin", user_email: "demo@example.invalid", user_role: "proxy_admin" };
const sessionExpires = Math.floor(Date.now() / 1000) + 12 * 60 * 60;
const jwtHeader = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString("base64url");
const session = {
  ...user,
  key: sessionKey,
  exp: sessionExpires,
  premium_user: false,
  login_method: "demo",
  password_reset_required: false,
  disabled_non_admin_personal_key_creation: false,
};
const jwtPayload = Buffer.from(JSON.stringify(session)).toString("base64url");
const jwtBody = `${jwtHeader}.${jwtPayload}`;
const sessionToken = `${jwtBody}.${createHmac("sha256", signingKey).update(jwtBody).digest("base64url")}`;

if (nextOrigin && (nextOrigin.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(nextOrigin.hostname))) {
  process.stderr.write("DEMO_NEXT_ORIGIN must be a local HTTP origin\n");
  process.exit(1);
}
if (nextOrigin?.port === String(port)) {
  process.stderr.write("Next and the preview server need different ports\n");
  process.exit(1);
}

const fixtures = {
  "/litellm/.well-known/litellm-ui-config": {
    server_root_path: "",
    proxy_base_url: previewOrigin,
    auto_redirect_to_sso: false,
    admin_ui_disabled: false,
    sso_configured: false,
    is_control_plane: false,
    workers: [],
  },
  "/ui/litellm-runtime-config.json": { backend_url: previewOrigin },
  "/litellm-runtime-config.json": { backend_url: previewOrigin },
  "/get/ui_settings": { values: {} },
  "/get/ui_theme_settings": { values: {} },
  "/public/litellm_blog_posts": { posts: [] },
  "/api/plugins": [],
  "/management/v1/users/me": user,
  "/health/readiness": { status: "healthy" },
  "/health/readiness/details": { status: "healthy", litellm_version: "demo", log_level: "INFO" },
  "/health/deployment_warnings": { ignore_available: false, warnings: [] },
  "/health/license": {
    has_license: false,
    license_type: null,
    expiration_date: null,
    allowed_features: [],
    limits: { max_users: null, max_teams: null },
  },
  "/user/available_users": {
    total_users: null,
    total_users_used: 0,
    total_users_remaining: null,
    total_teams: null,
    total_teams_used: 0,
    total_teams_remaining: null,
  },
  "/team/list": [],
  "/organization/list": [],
  "/v1/memory": { memories: [], total: 0 },
  "/sso/get/ui_settings": { PROXY_BASE_URL: previewOrigin, PROXY_LOGOUT_URL: `${previewOrigin}/preview` },
  "/get/user_banner": { enabled: false, message: "", severity: "info", revision: "demo" },
  "/get/latest_release_info": {
    version: "demo",
    release_url: "",
    new_features: 0,
    bug_fixes: 0,
    other_updates: 0,
  },
  "/telemetry/ui_events/enabled": { enabled: false },
  "/telemetry/settings": {
    available_destinations: [],
    destination: null,
    destination_set_by_environment: false,
    editable: false,
    environment_variables: [],
    flush_interval_seconds: 300,
    groups: [],
    report: null,
    report_is_sample: true,
    retention_days: 0,
    set_by_environment: false,
    stored_destination: null,
    stored_groups: [],
    vetoed: false,
  },
};
const publicFixtures = new Set([
  "/litellm/.well-known/litellm-ui-config",
  "/ui/litellm-runtime-config.json",
  "/litellm-runtime-config.json",
  "/get/ui_settings",
  "/get/ui_theme_settings",
  "/public/litellm_blog_posts",
  "/health/readiness",
]);
const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".webp": "image/webp",
};

function json(response, status, value) {
  response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(value));
}

function isAuthorized(request) {
  return request.headers.authorization === `Bearer ${sessionKey}` && Date.now() < sessionExpires * 1000;
}

function openPreview(response, search) {
  const destination = nextOrigin ? "/memory/demo/" : "/ui/memory/demo/";
  response.writeHead(302, {
    Location: `${destination}${search}`,
    "Set-Cookie": `token=${sessionToken}; Path=/; SameSite=Lax; Max-Age=43200`,
  });
  response.end();
}

async function serveFile(request, response, filename) {
  const details = await stat(filename).catch(() => null);
  if (!details?.isFile()) return false;
  response.writeHead(200, {
    "Content-Type": contentTypes[extname(filename)] ?? "application/octet-stream",
    "Content-Length": details.size,
  });
  if (request.method === "HEAD") response.end();
  else createReadStream(filename).pipe(response);
  return true;
}

async function serveExport(request, response, pathname) {
  const uiPath = pathname.replace(/^\/ui(?=\/|$)/, "").replace(/^\/litellm-asset-prefix(?=\/)/, "");
  const relative = uiPath.replace(/^\/+/, "");
  const exported = resolve(exportDir, relative);
  const publicAsset = resolve(publicDir, relative);
  const candidates = [exported, `${exported}.html`, resolve(exported, "index.html"), publicAsset];
  for (const candidate of candidates) {
    const insideExport = candidate.startsWith(`${exportDir}${sep}`);
    const insidePublic = candidate.startsWith(`${publicDir}${sep}`);
    if ((insideExport || insidePublic) && (await serveFile(request, response, candidate))) return;
  }
  json(response, 404, { error: "No exported page found. Build the dashboard or set DEMO_NEXT_ORIGIN." });
}

function proxyToNext(request, response, url) {
  const target = new URL(url.pathname.replace(/^\/ui(?=\/|$)/, "") + url.search, nextOrigin);
  const forwardedHeaders = Object.fromEntries(
    Object.entries(request.headers).filter(
      ([name]) => !["authorization", "cookie", "host", "connection"].includes(name),
    ),
  );
  const headers = { ...forwardedHeaders, "x-shared-knowledge-preview-proxy": "1" };
  const upstream = httpRequest(target, { method: request.method, headers }, (upstreamResponse) => {
    response.writeHead(upstreamResponse.statusCode ?? 502, upstreamResponse.headers);
    upstreamResponse.pipe(response);
  });
  upstream.on("error", () =>
    json(response, 502, { error: "Next.js is not ready. Start the local dev server and reload." }),
  );
  request.pipe(upstream);
}

function isPageOrAsset(request, url) {
  return (
    !request.headers.accept?.includes("application/json") &&
    /^\/(?:ui(?:\/|$)|memory(?:\/|$)|login(?:\/|$)|_next\/|__nextjs|litellm-asset-prefix\/|assets\/|favicon\.ico)/.test(
      url.pathname,
    )
  );
}

async function handleRequest(request, response) {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  if (request.headers["x-shared-knowledge-preview-proxy"] === "1") {
    json(response, 404, { error: "The requested route is missing from the local Next.js server." });
    return;
  }
  const url = new URL(request.url ?? "/", previewOrigin);
  const hostname = (request.headers.host ?? "").split(":")[0];
  if (!["127.0.0.1", "localhost"].includes(hostname)) {
    json(response, 403, { error: "This preview is local only." });
    return;
  }
  if (request.headers.origin) {
    const allowedOrigins = [previewOrigin, `http://localhost:${port}`, nextOrigin?.origin];
    if (!allowedOrigins.includes(request.headers.origin)) {
      json(response, 403, { error: "Only the local demo can use preview fixtures." });
      return;
    }
    response.setHeader("Access-Control-Allow-Origin", request.headers.origin);
    response.setHeader("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept");
    response.setHeader("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
  }
  if (request.method === "OPTIONS") {
    response.writeHead(204);
    response.end();
    return;
  }
  if (!["GET", "HEAD"].includes(request.method ?? "")) {
    json(response, 405, { error: "This preview serves read-only fixtures. Demo actions stay in the browser." });
    return;
  }
  if (["/", "/preview", "/preview/", "/knowledge-demo", "/knowledge-demo/"].includes(url.pathname)) {
    openPreview(response, url.search);
    return;
  }
  if (Object.hasOwn(fixtures, url.pathname)) {
    const permitted = publicFixtures.has(url.pathname) || isAuthorized(request);
    json(
      response,
      permitted ? 200 : 401,
      permitted ? fixtures[url.pathname] : { error: "Open /preview to start a demo session." },
    );
    return;
  }
  if (url.pathname === "/get_image") {
    await serveFile(request, response, resolve(proxyDir, "logo.png"));
    return;
  }
  if (url.pathname === "/get_favicon") {
    await serveFile(request, response, resolve(dashboardDir, "src/app/favicon.ico"));
    return;
  }
  if (!isPageOrAsset(request, url)) {
    json(response, 404, {
      error: "No fixture exists for this endpoint. The preview never contacts a production service.",
    });
    return;
  }
  if (nextOrigin) proxyToNext(request, response, url);
  else await serveExport(request, response, decodeURIComponent(url.pathname));
}

const server = createServer((request, response) => {
  void handleRequest(request, response).catch(() => {
    if (!response.headersSent) json(response, 500, { error: "The local preview could not serve this request." });
    else response.end();
  });
});
server.listen(port, "127.0.0.1", () => {
  process.stdout.write(`Local fixture preview ready at ${previewOrigin}/preview\n`);
});
