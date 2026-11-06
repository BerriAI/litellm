import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { fixtureRunSummaries, fixtureTraces, fixtureDefinitions } from "./data.ts";

const hostname = "127.0.0.1";
const port = 3378;
const dashboard = "http://workflow-runs.localhost:3377";
const user = { user_id: "fixture-user", user_email: "preview@example.test", user_role: "proxy_admin" };
const claims = {
  ...user,
  key: "fixture-only-not-a-gateway-key",
  login_method: "sso",
  exp: Math.floor(Date.now() / 1000) + 86400,
};
const token = `${Buffer.from('{"alg":"HS256","typ":"JWT"}').toString("base64url")}.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.fixture`;
const responses = {
  "/litellm/.well-known/litellm-ui-config": { server_root_path: "", proxy_base_url: dashboard },
  "/ui/litellm-runtime-config.json": {},
  "/litellm-runtime-config.json": {},
  "/management/v1/users/me": user,
  "/get/ui_theme_settings": { values: {} },
  "/get/ui_settings": { values: { enable_projects_ui: false } },
  "/sso/get/ui_settings": { values: { enable_projects_ui: false } },
  "/api/plugins": [],
  "/user/info": { user_id: user.user_id, user_info: { ...user, teams: [], models: [] }, teams: [], keys: [] },
  "/organization/info": [],
  "/team/list": [],
  "/health/readiness": { status: "healthy" },
  "/health/liveliness": { status: "healthy" },
  "/health/readiness/details": { status: "healthy" },
  "/health/deployment_warnings": { warnings: [], ignore_available: false },
  "/get/user_banner": { enabled: false, message: "", severity: "info", revision: "fixture" },
  "/user/available_users": [],
  "/telemetry/ui_events/enabled": { enabled: false },
  "/telemetry/settings": {
    groups: [],
    destination: "local_table",
    environment_variables: [],
    set_by_environment: false,
    vetoed: false,
  },
};

createServer((request, response) => {
  const url = new URL(request.url, dashboard);
  const pathname = url.pathname;
  const send = (payload) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(payload));
  };
  if (pathname === "/__workflow_run_fixture_login") {
    response.writeHead(302, {
      "Set-Cookie": `token=${token}; Path=/; SameSite=Lax; Max-Age=86400`,
      Location: `${dashboard}/ui/workflows/run-preview`,
    });
    response.end();
    return;
  }
  if (pathname === "/get_image") {
    response.writeHead(200, { "Content-Type": "image/png" });
    response.end(readFileSync(new URL("../../../../../litellm/proxy/logo.png", import.meta.url)));
    return;
  }
  if (pathname.startsWith("/v1/workflows/runs")) {
    if (request.headers.authorization !== `Bearer ${claims.key}`) {
      response.writeHead(403, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ error: "Fixture authentication required" }));
      return;
    }
    if (pathname === "/v1/workflows/runs") {
      const status = url.searchParams.get("status");
      const runs = fixtureRunSummaries.filter(
        (run) => run.workflow_id === url.searchParams.get("workflow_id") && (!status || run.state === status),
      );
      const after = url.searchParams.get("after");
      const start = after ? runs.findIndex((run) => run.run_id === after) + 1 : 0;
      const limit = Number(url.searchParams.get("limit") ?? 20);
      const page = runs.slice(start, start + limit);
      send({
        runs: page.map(({ state, ...run }) => ({ ...run, status: state })),
        next_cursor: start + limit < runs.length ? page.at(-1).run_id : null,
      });
      return;
    }
    const match = pathname.match(/^\/v1\/workflows\/runs\/([^/]+)\/(steps|definition)$/);
    const trace = match && fixtureTraces[decodeURIComponent(match[1])];
    if (trace) {
      send(match[2] === "steps" ? trace : fixtureDefinitions[trace.snapshot_id]);
      return;
    }
  }
  if (pathname.startsWith("/spend/logs/ui/fixture-request-")) {
    if (request.headers.authorization !== `Bearer ${claims.key}`) {
      response.writeHead(403, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ error: "Fixture authentication required" }));
      return;
    }
    const payload = {
      fixture: true,
      request_id: pathname.slice("/spend/logs/ui/".length),
      messages: [{ role: "user", content: "Summarize the release notes in one sentence." }],
      response: { content: "The release improves workflow debugging." },
    };
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(payload));
    return;
  }
  if (Object.hasOwn(responses, pathname)) {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(responses[pathname]));
    return;
  }
  process.stdout.write(`Unconfigured shell fixture ${request.method} ${pathname}\n`);
  response.writeHead(404, { "Content-Type": "application/json" });
  response.end(JSON.stringify({ error: "No fixture for this request" }));
}).listen(port, hostname, () => {
  process.stdout.write(`Fixture dashboard login ${dashboard}/__workflow_run_fixture_login\n`);
});
