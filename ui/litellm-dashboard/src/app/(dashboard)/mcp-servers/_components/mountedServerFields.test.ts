import { describe, expect, it } from "vitest";
import {
  mountedCreateFieldNames,
  mountedEditFieldNames,
  projectMountedCreateValues,
  projectMountedEditValues,
} from "./mountedServerFields";

const editRoot = (values: Record<string, unknown>) => mountedEditFieldNames(values).root;
const editCreds = (values: Record<string, unknown>) => mountedEditFieldNames(values).credentials;
const createRoot = (values: Record<string, unknown>) => mountedCreateFieldNames(values).root;
const createCreds = (values: Record<string, unknown>) => mountedCreateFieldNames(values).credentials;

const HTTP_NONE = { transport: "http", auth_type: "none" };

describe("edit root: transport gates", () => {
  it("mounts url for http but not spec_path or the stdio group", () => {
    const root = editRoot(HTTP_NONE);
    expect(root).toContain("url");
    expect(root).not.toContain("spec_path");
    expect(root).not.toContain("command");
    expect(root).not.toContain("stdio_config");
  });

  it("mounts url for sse", () => {
    expect(editRoot({ transport: "sse", auth_type: "none" })).toContain("url");
  });

  it("mounts spec_path and not url for openapi", () => {
    const root = editRoot({ transport: "openapi", auth_type: "none" });
    expect(root).toContain("spec_path");
    expect(root).not.toContain("url");
  });

  it("swaps the whole auth subtree for the stdio group on stdio", () => {
    const root = editRoot({ transport: "stdio", auth_type: "oauth2" });
    expect(root).toStrictEqual([
      "server_name",
      "alias",
      "description",
      "transport",
      "max_concurrent_requests",
      "command",
      "args",
      "env_json",
      "stdio_config",
      "env_vars",
      "allow_all_keys",
      "available_on_public_internet",
      "mcp_access_groups",
      "extra_headers",
      "static_headers",
    ]);
  });

  it("drops every credential on stdio even when auth_type is stored as oauth2", () => {
    expect(editCreds({ transport: "stdio", auth_type: "oauth2" })).toStrictEqual([]);
  });
});

describe("edit root: auth_type gates", () => {
  it("mounts credentials.auth_value only for the four value-bearing auth types", () => {
    for (const authType of ["api_key", "bearer_token", "token", "basic"]) {
      expect(editCreds({ transport: "http", auth_type: authType })).toStrictEqual(["auth_value"]);
    }
    expect(editCreds(HTTP_NONE)).toStrictEqual([]);
  });

  it("swaps the oauth2 endpoint set on the M2M flow", () => {
    const m2m = editRoot({ transport: "http", auth_type: "oauth2", oauth_flow_type: "m2m" });
    const interactive = editRoot({ transport: "http", auth_type: "oauth2", oauth_flow_type: "interactive" });
    expect(m2m).toContain("token_url");
    expect(m2m).not.toContain("issuer");
    expect(m2m).not.toContain("registration_url");
    expect(interactive).toContain("issuer");
    expect(interactive).toContain("registration_url");
  });

  it("mounts token_validation_json ONLY on the interactive oauth2 branch", () => {
    expect(editRoot({ transport: "http", auth_type: "oauth2", oauth_flow_type: "interactive" })).toContain(
      "token_validation_json",
    );
    expect(editRoot({ transport: "http", auth_type: "oauth2", oauth_flow_type: "m2m" })).not.toContain(
      "token_validation_json",
    );
    expect(editRoot({ transport: "http", auth_type: "oauth2_token_exchange" })).not.toContain("token_validation_json");
    expect(editRoot(HTTP_NONE)).not.toContain("token_validation_json");
  });

  it("mounts token_storage_ttl_seconds only on the interactive oauth2 branch", () => {
    expect(editRoot({ transport: "http", auth_type: "oauth2", oauth_flow_type: "interactive" })).toContain(
      "token_storage_ttl_seconds",
    );
    expect(editRoot({ transport: "http", auth_type: "oauth2", oauth_flow_type: "m2m" })).not.toContain(
      "token_storage_ttl_seconds",
    );
  });

  it("gates audience and subject_token_type on the entra_obo token-exchange profile", () => {
    const rfc = editRoot({ transport: "http", auth_type: "oauth2_token_exchange", token_exchange_profile: "rfc8693" });
    const entra = editRoot({
      transport: "http",
      auth_type: "oauth2_token_exchange",
      token_exchange_profile: "entra_obo",
    });
    expect(rfc).toContain("audience");
    expect(rfc).toContain("subject_token_type");
    expect(entra).not.toContain("audience");
    expect(entra).not.toContain("subject_token_type");
    expect(entra).toContain("token_exchange_profile");
  });

  it("mounts the seven aws credentials only for aws_sigv4", () => {
    expect(sorted(editCreds({ transport: "http", auth_type: "aws_sigv4" }))).toStrictEqual(
      sorted([
        "aws_region_name",
        "aws_service_name",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_role_name",
        "aws_session_name",
      ]),
    );
    expect(editCreds(HTTP_NONE)).not.toContain("aws_region_name");
  });

  it("mounts the id-jag credential set only for oauth2_id_jag", () => {
    const creds = editCreds({ transport: "http", auth_type: "oauth2_id_jag" });
    expect(creds).toContain("id_jag_resource_token_endpoint");
    expect(creds).toContain("client_private_key");
    expect(creds).toContain("client_assertion_signing_alg");
    expect(editCreds(HTTP_NONE)).not.toContain("id_jag_resource_token_endpoint");
  });
});

describe("edit root: children that gate by early return null", () => {
  it("mounts dcr_bridge and the declared-app credentials only for the client-forwarded modes", () => {
    for (const authType of ["true_passthrough", "oauth_delegate"]) {
      expect(editRoot({ transport: "http", auth_type: authType })).toContain("dcr_bridge");
      expect(sorted(editCreds({ transport: "http", auth_type: authType }))).toStrictEqual(
        sorted(["client_id", "client_secret"]),
      );
    }
    expect(editRoot(HTTP_NONE)).not.toContain("dcr_bridge");
    expect(editRoot({ transport: "http", auth_type: "oauth2" })).not.toContain("dcr_bridge");
  });

  it("unmounts dcr_bridge with its parent section on stdio", () => {
    expect(editRoot({ transport: "stdio", auth_type: "true_passthrough" })).not.toContain("dcr_bridge");
  });
});

describe("edit root: permission-section gates", () => {
  it("mounts delegate_auth_to_upstream only for oauth2", () => {
    expect(editRoot({ transport: "http", auth_type: "oauth2" })).toContain("delegate_auth_to_upstream");
    expect(editRoot(HTTP_NONE)).not.toContain("delegate_auth_to_upstream");
    expect(editRoot({ transport: "http", auth_type: "api_key" })).not.toContain("delegate_auth_to_upstream");
  });

  it("mounts oauth_passthrough only for none-auth WITH an Authorization extra header", () => {
    expect(editRoot({ ...HTTP_NONE, extra_headers: ["Authorization"] })).toContain("oauth_passthrough");
    expect(editRoot({ ...HTTP_NONE, extra_headers: ["authorization"] })).toContain("oauth_passthrough");
    expect(editRoot({ ...HTTP_NONE, extra_headers: ["X-Other"] })).not.toContain("oauth_passthrough");
    expect(editRoot(HTTP_NONE)).not.toContain("oauth_passthrough");
    expect(editRoot({ transport: "http", auth_type: "oauth2", extra_headers: ["Authorization"] })).not.toContain(
      "oauth_passthrough",
    );
  });

  it("treats an absent auth_type as none-auth for the oauth_passthrough gate", () => {
    expect(editRoot({ transport: "http", extra_headers: ["Authorization"] })).toContain("oauth_passthrough");
  });
});

describe("create root: where it diverges from edit", () => {
  it("mounts source_url, which the edit root has no binding for", () => {
    expect(createRoot({ transport: "http", auth_type: "none" })).toContain("source_url");
    expect(editRoot(HTTP_NONE)).not.toContain("source_url");
  });

  it("gates url on an allow-list, so a blank transport mounts NEITHER url nor auth_type", () => {
    const blank = createRoot({ transport: "" });
    expect(blank).not.toContain("url");
    expect(blank).not.toContain("auth_type");
    expect(editRoot({ transport: "" })).toContain("url");
    expect(editRoot({ transport: "" })).toContain("auth_type");
  });

  it("mounts stdio_config on stdio but never the edit root's command/args/env_json", () => {
    const root = createRoot({ transport: "stdio" });
    expect(root).toContain("stdio_config");
    expect(root).not.toContain("command");
    expect(root).not.toContain("args");
    expect(root).not.toContain("env_json");
  });

  it("mounts the byok fields only for openapi with is_byok on", () => {
    expect(createRoot({ transport: "openapi", auth_type: "none" })).toContain("is_byok");
    expect(createRoot({ transport: "openapi", auth_type: "none" })).not.toContain("byok_description");
    const on = createRoot({ transport: "openapi", auth_type: "none", is_byok: true });
    expect(on).toContain("byok_description");
    expect(on).toContain("byok_api_key_help_url");
    expect(createRoot({ transport: "http", auth_type: "none", is_byok: true })).not.toContain("byok_description");
  });

  it("drops every credential while the transport is unset", () => {
    expect(createCreds({ transport: "", auth_type: "aws_sigv4" })).toStrictEqual([]);
    expect(createCreds({ transport: "http", auth_type: "aws_sigv4" })).toContain("aws_region_name");
  });
});

const ALWAYS = ["server_name", "alias", "description", "transport", "max_concurrent_requests"];
const PERMS = [
  "allow_all_keys",
  "available_on_public_internet",
  "mcp_access_groups",
  "extra_headers",
  "static_headers",
];
const sorted = (xs: readonly string[]) => [...xs].sort();

const expectEditSets = (
  values: Record<string, unknown>,
  expected: { root: readonly string[]; credentials: readonly string[] },
) => {
  expect(sorted(editRoot(values))).toStrictEqual(sorted(expected.root));
  expect(sorted(editCreds(values))).toStrictEqual(sorted(expected.credentials));
};

const expectCreateSets = (
  values: Record<string, unknown>,
  expected: { root: readonly string[]; credentials: readonly string[] },
) => {
  expect(sorted(createRoot(values))).toStrictEqual(sorted(expected.root));
  expect(sorted(createCreds(values))).toStrictEqual(sorted(expected.credentials));
};

describe("edit root: exact mounted set per auth configuration", () => {
  it("http + none", () => {
    expectEditSets(HTTP_NONE, { root: [...ALWAYS, "url", "auth_type", "env_vars", ...PERMS], credentials: [] });
  });

  it("http + api_key", () => {
    expectEditSets(
      { transport: "http", auth_type: "api_key" },
      { root: [...ALWAYS, "url", "auth_type", "env_vars", ...PERMS], credentials: ["auth_value"] },
    );
  });

  it("http + oauth2 M2M", () => {
    expectEditSets(
      { transport: "http", auth_type: "oauth2", oauth_flow_type: "m2m" },
      {
        root: [
          ...ALWAYS,
          "url",
          "auth_type",
          "oauth_flow_type",
          "token_url",
          "env_vars",
          ...PERMS,
          "delegate_auth_to_upstream",
        ],
        credentials: ["client_id", "client_secret", "token_endpoint_auth_method", "scopes", "upstream_resource"],
      },
    );
  });

  it("http + oauth2 interactive", () => {
    expectEditSets(
      { transport: "http", auth_type: "oauth2", oauth_flow_type: "interactive" },
      {
        root: [
          ...ALWAYS,
          "url",
          "auth_type",
          "oauth_flow_type",
          "issuer",
          "authorization_url",
          "token_url",
          "registration_url",
          "token_validation_json",
          "token_storage_ttl_seconds",
          "env_vars",
          ...PERMS,
          "delegate_auth_to_upstream",
        ],
        credentials: ["client_id", "client_secret", "scopes", "upstream_resource", "token_endpoint_auth_method"],
      },
    );
  });

  it("http + token exchange, rfc8693", () => {
    expectEditSets(
      { transport: "http", auth_type: "oauth2_token_exchange", token_exchange_profile: "rfc8693" },
      {
        root: [
          ...ALWAYS,
          "url",
          "auth_type",
          "token_exchange_profile",
          "token_exchange_endpoint",
          "audience",
          "subject_token_type",
          "env_vars",
          ...PERMS,
        ],
        credentials: ["client_id", "client_secret", "scopes"],
      },
    );
  });

  it("http + token exchange, entra_obo keeps the endpoint while dropping audience and subject_token_type", () => {
    expectEditSets(
      { transport: "http", auth_type: "oauth2_token_exchange", token_exchange_profile: "entra_obo" },
      {
        root: [
          ...ALWAYS,
          "url",
          "auth_type",
          "token_exchange_profile",
          "token_exchange_endpoint",
          "env_vars",
          ...PERMS,
        ],
        credentials: ["client_id", "client_secret", "scopes"],
      },
    );
  });

  it("http + id-jag", () => {
    expectEditSets(
      { transport: "http", auth_type: "oauth2_id_jag" },
      {
        root: [
          ...ALWAYS,
          "url",
          "auth_type",
          "token_exchange_endpoint",
          "audience",
          "subject_token_type",
          "env_vars",
          ...PERMS,
        ],
        credentials: [
          "id_jag_resource_token_endpoint",
          "client_id",
          "client_secret",
          "client_private_key",
          "client_private_key_id",
          "client_assertion_signing_alg",
          "id_jag_resource",
          "scopes",
        ],
      },
    );
  });

  it("http + true_passthrough", () => {
    expectEditSets(
      { transport: "http", auth_type: "true_passthrough" },
      {
        root: [...ALWAYS, "url", "auth_type", "dcr_bridge", "env_vars", ...PERMS],
        credentials: ["client_id", "client_secret"],
      },
    );
  });

  it("openapi + none", () => {
    expectEditSets(
      { transport: "openapi", auth_type: "none" },
      { root: [...ALWAYS, "spec_path", "auth_type", "env_vars", ...PERMS], credentials: [] },
    );
  });
});

describe("create root: exact mounted set per configuration", () => {
  it("http + none", () => {
    expectCreateSets(
      { transport: "http", auth_type: "none" },
      { root: [...ALWAYS, "source_url", "url", "auth_type", "env_vars", ...PERMS], credentials: [] },
    );
  });

  it("openapi with byok on", () => {
    expectCreateSets(
      { transport: "openapi", auth_type: "none", is_byok: true },
      {
        root: [
          ...ALWAYS,
          "source_url",
          "spec_path",
          "is_byok",
          "byok_description",
          "byok_api_key_help_url",
          "auth_type",
          "env_vars",
          ...PERMS,
        ],
        credentials: [],
      },
    );
  });

  it("stdio", () => {
    expectCreateSets(
      { transport: "stdio" },
      { root: [...ALWAYS, "source_url", "stdio_config", "env_vars", ...PERMS], credentials: [] },
    );
  });

  it("transport still unset", () => {
    expectCreateSets({ transport: "" }, { root: [...ALWAYS, "source_url", "env_vars", ...PERMS], credentials: [] });
  });

  it("http + oauth2 interactive", () => {
    expectCreateSets(
      { transport: "http", auth_type: "oauth2", oauth_flow_type: "interactive" },
      {
        root: [
          ...ALWAYS,
          "source_url",
          "url",
          "auth_type",
          "oauth_flow_type",
          "issuer",
          "authorization_url",
          "token_url",
          "registration_url",
          "token_validation_json",
          "token_storage_ttl_seconds",
          "env_vars",
          ...PERMS,
          "delegate_auth_to_upstream",
        ],
        credentials: ["client_id", "client_secret", "scopes", "upstream_resource", "token_endpoint_auth_method"],
      },
    );
  });

  it("http + oauth_delegate mounts dcr_bridge and the declared app", () => {
    expectCreateSets(
      { transport: "http", auth_type: "oauth_delegate" },
      {
        root: [...ALWAYS, "source_url", "url", "auth_type", "dcr_bridge", "env_vars", ...PERMS],
        credentials: ["client_id", "client_secret"],
      },
    );
  });
});

describe("projection shape", () => {
  it("EMITS a mounted-but-unset field as a key holding undefined, matching antd onFinish", () => {
    const projected = projectMountedEditValues({ transport: "http", auth_type: "none", server_name: "s" });
    expect("description" in projected).toBe(true);
    expect(projected.description).toBeUndefined();
    expect(Object.keys(projected)).toContain("max_concurrent_requests");
  });

  it("emits mounted-but-unset CREDENTIAL keys as undefined rather than omitting them", () => {
    const projected = projectMountedEditValues({ transport: "http", auth_type: "api_key" });
    expect(Object.keys(projected.credentials as object)).toStrictEqual(["auth_value"]);
    expect((projected.credentials as Record<string, unknown>).auth_value).toBeUndefined();
  });

  it("omits the credentials key entirely when no credential field is mounted", () => {
    expect("credentials" in projectMountedEditValues(HTTP_NONE)).toBe(false);
  });

  it("drops an unmounted field even when the store still holds a value for it", () => {
    const storeWithStaleHttpValues = {
      transport: "stdio",
      auth_type: "oauth2",
      url: "https://kept-in-store.example",
      issuer: "https://kept-in-store.example",
      command: "npx",
    };
    const projected = projectMountedEditValues(storeWithStaleHttpValues);
    expect("url" in projected).toBe(false);
    expect("issuer" in projected).toBe(false);
    expect(projected.command).toBe("npx");
  });

  it("passes list rows through whole, since a list field is projected as one key and not per mounted sub-field", () => {
    const row = { name: "N", value: "V", scope: "user", description: "D" };
    const projected = projectMountedEditValues({ ...HTTP_NONE, env_vars: [row] });
    expect(projected.env_vars).toStrictEqual([row]);
  });

  it("keeps static_headers rows whole", () => {
    const rows = [{ header: "X-A", value: "1" }];
    expect(
      projectMountedCreateValues({ transport: "http", auth_type: "none", static_headers: rows }).static_headers,
    ).toStrictEqual(rows);
  });
});
