import { createServer } from "node:http";

const hostname = "127.0.0.1";
const port = 3368;
const dashboard = "http://workflow-graph.localhost:3367";
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
  "/api/plugins": [],
  "/user/info": { user_id: user.user_id, user_info: { ...user, teams: [], models: [] }, teams: [], keys: [] },
  "/organization/info": [],
  "/team/list": [],
  "/sso/get/ui_settings": {},
  "/health/readiness/details": { status: "healthy" },
  "/health/deployment_warnings": { warnings: [], ignore_available: false },
  "/health/license": { has_license: false, allowed_features: [], limits: { max_users: null, max_teams: null } },
  "/user/available_users": { total_users_used: 1, total_teams_used: 0 },
  "/get/user_banner": { enabled: false, message: "", severity: "info" },
  "/get/latest_release_info": null,
  "/model/deprecations": { deprecated: [], imminent: [], upcoming: [] },
  "/telemetry/settings": { environment_variables: [], groups: [] },
  "/telemetry/ui_events/enabled": { enabled: false },
  "/health/readiness": { status: "healthy" },
  "/health/liveliness": { status: "healthy" },
};

createServer((request, response) => {
  const pathname = new URL(request.url, dashboard).pathname;
  if (pathname === "/__workflow_fixture_login") {
    response.writeHead(302, {
      "Set-Cookie": `token=${token}; Path=/; SameSite=Lax; Max-Age=86400`,
      Location: `${dashboard}/workflows/graph-preview`,
    });
    response.end();
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
  process.stdout.write(`Fixture dashboard login ${dashboard}/__workflow_fixture_login\n`);
});
