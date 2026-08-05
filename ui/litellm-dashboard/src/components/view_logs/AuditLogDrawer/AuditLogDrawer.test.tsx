import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { AuditLogEntry } from "../AuditLogsTableColumns";
import { AuditLogDrawer } from "./AuditLogDrawer";

const BASE_LOG: AuditLogEntry = {
  id: "log-1",
  updated_at: "2026-07-20T12:00:00Z",
  changed_by: "user-42",
  changed_by_api_key: "sk-hash-abc",
  action: "updated",
  table_name: "LiteLLM_TeamTable",
  object_id: "team-obj-123",
  before_value: { spend: 1 },
  updated_values: { spend: 2 },
};

const ENRICHED_LOG: AuditLogEntry = {
  ...BASE_LOG,
  object_alias: "prod-team",
  changed_by_user_email: "admin@example.com",
  changed_by_key_alias: "admin-key",
};

function renderDrawer(log: AuditLogEntry) {
  render(<AuditLogDrawer open={true} onClose={vi.fn()} log={log} />);
}

describe("AuditLogDrawer", () => {
  it("shows the alias, changed-by email, and key alias alongside the raw id and hash", () => {
    renderDrawer(ENRICHED_LOG);

    expect(screen.getByText("prod-team")).toBeInTheDocument();
    expect(screen.getByText("admin@example.com")).toBeInTheDocument();
    expect(screen.getByText("user-42")).toBeInTheDocument();
    expect(screen.getByText("admin-key")).toBeInTheDocument();
    expect(screen.getByText("sk-hash-abc")).toBeInTheDocument();
  });

  it("falls back gracefully when the enrichment fields are absent", () => {
    renderDrawer(BASE_LOG);

    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("user-42")).toBeInTheDocument();
    expect(screen.getByText("sk-hash-abc")).toBeInTheDocument();
    expect(screen.queryByText("admin@example.com")).toBeNull();
  });
});
