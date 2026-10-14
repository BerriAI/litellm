import { describe, expect, it } from "vitest";

import type { MCPServer } from "@/components/mcp_tools/types";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import {
  buildBulkCustomerPayload,
  emptyBulkCustomerEdits,
  MAX_BULK_CUSTOMERS,
  type BulkCustomerEdits,
} from "./bulkCustomerPayload";

const customers: EndUser[] = [
  { user_id: "customer-a", alias: "Original A", blocked: true, spend: 0, models: ["old-model-a"] },
  { user_id: "customer-b", alias: "Original B", blocked: false, spend: 0, models: ["old-model-b"] },
];

describe("bulk customer edits", () => {
  it("rejects a request with no opted-in fields even when hidden values are populated", () => {
    const edits: BulkCustomerEdits = {
      budget: { enabled: false, mode: "max", budgetId: "old-budget", maxBudget: "100" },
      models: { enabled: false, value: ["new-model"] },
      region: { enabled: false, value: "eu" },
      defaultModel: { enabled: false, value: "new-default" },
      mcpServers: { enabled: false, value: ["new-server"] },
      mcpAccessGroups: { enabled: false, value: ["new-group"] },
      mcpToolsets: { enabled: false, value: ["new-toolset"] },
    };
    expect(buildBulkCustomerPayload(customers, edits, [], customers)).toEqual({
      ok: false,
      error: "Choose at least one field to update",
    });
  });

  it("sends only toolsets and customer IDs when only toolsets are opted in", () => {
    const edits = { ...emptyBulkCustomerEdits(), mcpToolsets: { enabled: true, value: ["new-toolset"] } };
    expect(buildBulkCustomerPayload(customers, edits, [], customers)).toEqual({
      ok: true,
      payload: {
        customers: [
          { user_id: "customer-a", object_permission: { mcp_toolsets: ["new-toolset"] } },
          { user_id: "customer-b", object_permission: { mcp_toolsets: ["new-toolset"] } },
        ],
      },
    });
  });

  it.each([
    ["models", "models"],
    ["mcpServers", "mcp_servers"],
    ["mcpAccessGroups", "mcp_access_groups"],
    ["mcpToolsets", "mcp_toolsets"],
  ] as const)("clears only the opted-in %s field with an explicit empty array", (field, apiField) => {
    const edits = { ...emptyBulkCustomerEdits(), [field]: { enabled: true, value: [] } };
    const update =
      field === "models"
        ? { models: [] }
        : { object_permission: { [apiField]: [], ...(field === "mcpServers" ? { mcp_tool_permissions: {} } : {}) } };
    expect(buildBulkCustomerPayload(customers, edits, [], customers)).toEqual({
      ok: true,
      payload: { customers: customers.map((customer) => ({ user_id: customer.user_id, ...update })) },
    });
  });

  it("omits unchecked MCP fields even when their values are nonempty", () => {
    const edits = {
      ...emptyBulkCustomerEdits(),
      mcpServers: { enabled: false, value: ["hidden-server"] },
      mcpAccessGroups: { enabled: false, value: ["hidden-group"] },
      mcpToolsets: { enabled: true, value: ["replacement-toolset"] },
    };
    expect(buildBulkCustomerPayload([customers[0]], edits, [], customers)).toEqual({
      ok: true,
      payload: { customers: [{ user_id: "customer-a", object_permission: { mcp_toolsets: ["replacement-toolset"] } }] },
    });
  });

  it("applies independently checked fields without alias or blocking changes", () => {
    const edits: BulkCustomerEdits = {
      budget: { enabled: true, mode: "link", budgetId: "replacement-budget", maxBudget: "900" },
      models: { enabled: true, value: ["model-a", "model-b"] },
      region: { enabled: true, value: "us" },
      defaultModel: { enabled: true, value: " model-a " },
      mcpServers: { enabled: true, value: ["server-a"] },
      mcpAccessGroups: { enabled: true, value: ["group-a"] },
      mcpToolsets: { enabled: true, value: ["toolset-a"] },
    };
    expect(buildBulkCustomerPayload([customers[0]], edits, [], customers)).toEqual({
      ok: true,
      payload: {
        customers: [
          {
            user_id: "customer-a",
            budget_id: "replacement-budget",
            models: ["model-a", "model-b"],
            allowed_model_region: "us",
            default_model: "model-a",
            object_permission: {
              mcp_servers: ["server-a"],
              mcp_tool_permissions: {},
              mcp_access_groups: ["group-a"],
              mcp_toolsets: ["toolset-a"],
            },
          },
        ],
      },
    });
  });

  it("prunes removed servers per customer and preserves the retained server's tool restrictions", () => {
    const servers: MCPServer[] = [
      {
        server_id: "kept",
        server_name: "kept-name",
        alias: "shared",
        created_at: "",
        created_by: "",
        updated_at: "",
        updated_by: "",
      },
      {
        server_id: "removed",
        server_name: "removed-name",
        alias: "shared",
        created_at: "",
        created_by: "",
        updated_at: "",
        updated_by: "",
      },
    ];
    const rows = customers.map((customer, index) => ({
      ...customer,
      object_permission: {
        mcp_servers: ["kept", "removed"],
        mcp_tool_permissions: {
          kept: [`read-${index}`],
          "kept-name": [`write-${index}`],
          shared: [`list-${index}`],
          removed: ["delete"],
          "removed-name": ["admin"],
        },
      },
    }));
    const edits = { ...emptyBulkCustomerEdits(), mcpServers: { enabled: true, value: ["kept"] } };
    expect(buildBulkCustomerPayload(rows, edits, servers, customers)).toEqual({
      ok: true,
      payload: {
        customers: rows.map((customer, index) => ({
          user_id: customer.user_id,
          object_permission: {
            mcp_servers: ["kept"],
            mcp_tool_permissions: { kept: [`read-${index}`, `write-${index}`, `list-${index}`] },
          },
        })),
      },
    });
  });

  it.each([{ tools: [] }, { tools: ["*"] }])(
    "matches IDs before aliases while retaining explicit tool restrictions $tools",
    ({ tools }) => {
      const servers: MCPServer[] = [
        { server_id: "kept", alias: "removed", created_at: "", created_by: "", updated_at: "", updated_by: "" },
        { server_id: "removed", alias: "kept", created_at: "", created_by: "", updated_at: "", updated_by: "" },
      ];
      const customer = {
        ...customers[0],
        object_permission: { mcp_tool_permissions: { kept: tools, removed: ["admin"] } },
      };
      expect(
        buildBulkCustomerPayload(
          [customer],
          {
            ...emptyBulkCustomerEdits(),
            mcpServers: { enabled: true, value: ["kept"] },
          },
          servers,
          customers,
        ),
      ).toEqual({
        ok: true,
        payload: {
          customers: [
            {
              user_id: customer.user_id,
              object_permission: {
                mcp_servers: ["kept"],
                mcp_tool_permissions: { kept: tools },
              },
            },
          ],
        },
      });
    },
  );

  it("clears existing tool grants when the server replacement is empty", () => {
    const customer = { ...customers[0], object_permission: { mcp_tool_permissions: { removed: ["read"] } } };
    expect(
      buildBulkCustomerPayload(
        [customer],
        {
          ...emptyBulkCustomerEdits(),
          mcpServers: { enabled: true, value: [] },
        },
        [],
        customers,
      ),
    ).toEqual({
      ok: true,
      payload: {
        customers: [
          {
            user_id: customer.user_id,
            object_permission: {
              mcp_servers: [],
              mcp_tool_permissions: {},
            },
          },
        ],
      },
    });
  });

  it("sends only max_budget in maximum mode even with a hidden budget link", () => {
    const edits = {
      ...emptyBulkCustomerEdits(),
      budget: { enabled: true, mode: "max" as const, budgetId: "hidden-budget", maxBudget: "1.5" },
    };
    expect(buildBulkCustomerPayload(customers, edits, [], customers)).toEqual({
      ok: true,
      payload: {
        customers: [
          { user_id: "customer-a", max_budget: 1.5 },
          { user_id: "customer-b", max_budget: 1.5 },
        ],
      },
    });
  });

  it.each(["", "0", "-1", "Infinity", "NaN", "not-a-number"])("rejects maximum budget %j", (maxBudget) => {
    const edits = {
      ...emptyBulkCustomerEdits(),
      budget: { enabled: true, mode: "max" as const, budgetId: "", maxBudget },
    };
    expect(buildBulkCustomerPayload(customers, edits, [], customers)).toEqual({
      ok: false,
      error: "Enter a maximum budget greater than zero",
    });
  });

  it("rejects a maximum edit when a budget is shared with an unselected customer", () => {
    const edits = {
      ...emptyBulkCustomerEdits(),
      budget: { enabled: true, mode: "max" as const, budgetId: "", maxBudget: "100" },
    };
    expect(
      buildBulkCustomerPayload(
        [{ ...customers[0], budget_id: "shared-budget" }],
        edits,
        [],
        customers.map((customer) => ({ ...customer, budget_id: "shared-budget" })),
      ),
    ).toEqual({
      ok: false,
      error: "Max Budget cannot be changed for customers with shared budgets. Select an existing budget instead",
    });
  });

  it("updates each customer's own budget maximum without replacing budget links", () => {
    const rows = customers.map((customer) => ({ ...customer, budget_id: `own-${customer.user_id}` }));
    expect(
      buildBulkCustomerPayload(
        rows,
        {
          ...emptyBulkCustomerEdits(),
          budget: { enabled: true, mode: "max", budgetId: "", maxBudget: "75" },
        },
        [],
        rows,
      ),
    ).toEqual({
      ok: true,
      payload: { customers: rows.map((customer) => ({ user_id: customer.user_id, max_budget: 75 })) },
    });
  });

  it("can replace an existing budget link without changing its maximum", () => {
    const edits = {
      ...emptyBulkCustomerEdits(),
      budget: { enabled: true, mode: "link" as const, budgetId: "new-budget", maxBudget: "" },
    };
    expect(buildBulkCustomerPayload([{ ...customers[0], budget_id: "shared-budget" }], edits, [], customers)).toEqual({
      ok: true,
      payload: { customers: [{ user_id: "customer-a", budget_id: "new-budget" }] },
    });
  });

  it("rejects clearing a budget link", () => {
    expect(
      buildBulkCustomerPayload(
        customers,
        {
          ...emptyBulkCustomerEdits(),
          budget: { enabled: true, mode: "link", budgetId: "", maxBudget: "" },
        },
        [],
        customers,
      ),
    ).toEqual({
      ok: false,
      error: "Select an existing budget",
    });
  });

  it("rejects clearing an allowed region", () => {
    expect(
      buildBulkCustomerPayload(
        customers,
        { ...emptyBulkCustomerEdits(), region: { enabled: true, value: "" } },
        [],
        customers,
      ),
    ).toEqual({
      ok: false,
      error: "Choose EU or US for the allowed region",
    });
  });

  it("rejects clearing a default model", () => {
    expect(
      buildBulkCustomerPayload(
        customers,
        { ...emptyBulkCustomerEdits(), defaultModel: { enabled: true, value: " " } },
        [],
        customers,
      ),
    ).toEqual({
      ok: false,
      error: "Select a default model",
    });
  });

  it("rejects empty customer selection", () => {
    expect(
      buildBulkCustomerPayload(
        [],
        { ...emptyBulkCustomerEdits(), models: { enabled: true, value: [] } },
        [],
        customers,
      ),
    ).toEqual({
      ok: false,
      error: "Select at least one customer",
    });
  });

  it("accepts the endpoint's maximum batch size", () => {
    const batch = Array.from({ length: MAX_BULK_CUSTOMERS }, (_, index) => ({
      ...customers[0],
      user_id: `customer-${index}`,
    }));
    const result = buildBulkCustomerPayload(
      batch,
      {
        ...emptyBulkCustomerEdits(),
        models: { enabled: true, value: [] },
      },
      [],
      customers,
    );
    expect(result).toEqual({
      ok: true,
      payload: { customers: batch.map((customer) => ({ user_id: customer.user_id, models: [] })) },
    });
  });

  it("rejects a selection larger than the endpoint's batch limit", () => {
    const batch = Array.from({ length: MAX_BULK_CUSTOMERS + 1 }, (_, index) => ({
      ...customers[0],
      user_id: `customer-${index}`,
    }));
    expect(
      buildBulkCustomerPayload(
        batch,
        { ...emptyBulkCustomerEdits(), models: { enabled: true, value: [] } },
        [],
        customers,
      ),
    ).toEqual({
      ok: false,
      error: `Select at most ${MAX_BULK_CUSTOMERS} customers`,
    });
  });
});
