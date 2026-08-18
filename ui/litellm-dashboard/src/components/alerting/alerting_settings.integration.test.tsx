import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";

const toastMock = vi.hoisted(() => ({
  success: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  error: vi.fn(),
  fromError: vi.fn(),
  dismiss: vi.fn(),
}));
vi.mock("@/lib/toast", () => ({ toast: toastMock }));

import AlertingSettings from "./alerting_settings";

const PROXY_SETTINGS = [
  {
    field_name: "slack_alerting",
    field_type: "Boolean",
    field_value: false,
    field_default_value: null,
    field_description: "Enable slack alerting",
    stored_in_db: false,
    premium_field: false,
  },
  {
    field_name: "daily_report_frequency",
    field_type: "Integer",
    field_value: 43200,
    field_default_value: 43200,
    field_description: "Frequency of deployment reports",
    stored_in_db: false,
    premium_field: false,
  },
  {
    field_name: "budget_alert_ttl",
    field_type: "Integer",
    field_value: 86400,
    field_default_value: 86400,
    field_description: "Cache ttl for budget alerts",
    stored_in_db: false,
    premium_field: false,
  },
];

interface Captured {
  readonly url: string;
  readonly body: string | undefined;
}

const installProxy = (settings: unknown, updateStatus = 200): Captured[] => {
  const calls: Captured[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: unknown, init?: RequestInit) => {
      calls.push({ url: String(url), body: init?.body as string | undefined });
      if (!init?.method || init.method === "GET") {
        return new Response(JSON.stringify(settings), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(
        JSON.stringify(updateStatus === 200 ? { status: "success" } : { detail: { error: "rejected by proxy" } }),
        { status: updateStatus, headers: { "Content-Type": "application/json" } },
      );
    }),
  );
  return calls;
};

const alertingArgsBody = (calls: readonly Captured[]): Record<string, unknown> | undefined => {
  const write = calls
    .filter((call) => call.url.includes("/config/field/update"))
    .map((call) => JSON.parse(call.body ?? "{}") as { field_name?: string; field_value?: Record<string, unknown> })
    .find((payload) => payload.field_name === "alerting_args");
  return write?.field_value;
};

const renderPage = async (settings: unknown, updateStatus = 200) => {
  const calls = installProxy(settings, updateStatus);
  const user = userEvent.setup();
  render(<AlertingSettings accessToken="sk-test" premiumUser={true} />);
  await screen.findByText("daily_report_frequency");
  return { calls, user };
};

const save = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("button", { name: "Update Settings" }));
  await waitFor(() =>
    expect(toastMock.success.mock.calls.length + toastMock.fromError.mock.calls.length).toBeGreaterThan(0),
  );
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("AlertingSettings alerting_args payload", () => {
  it("omits an Integer field the user cleared instead of writing null into stored config", async () => {
    const { calls, user } = await renderPage(PROXY_SETTINGS);

    await user.clear(screen.getByDisplayValue("43200"));
    await user.click(screen.getByRole("switch"));
    await save(user);

    const args = alertingArgsBody(calls);
    expect(args).toBeDefined();
    expect(args).not.toHaveProperty("daily_report_frequency");
  });

  it("still sends the Integer fields the user did not clear", async () => {
    const { calls, user } = await renderPage(PROXY_SETTINGS);

    await user.clear(screen.getByDisplayValue("43200"));
    await user.click(screen.getByRole("switch"));
    await save(user);

    expect(alertingArgsBody(calls)).toEqual({ budget_alert_ttl: 86400 });
  });

  it("sends an edited Integer field as a number", async () => {
    const { calls, user } = await renderPage(PROXY_SETTINGS);

    await user.clear(screen.getByDisplayValue("43200"));
    await user.type(screen.getByDisplayValue(""), "600");
    await save(user);

    expect(alertingArgsBody(calls)).toEqual({ daily_report_frequency: 600, budget_alert_ttl: 86400 });
  });

  it("sends an Integer field the user set to zero", async () => {
    const { calls, user } = await renderPage(PROXY_SETTINGS);

    await user.clear(screen.getByDisplayValue("43200"));
    await user.type(screen.getByDisplayValue(""), "0");
    await save(user);

    expect(alertingArgsBody(calls)).toEqual({ daily_report_frequency: 0, budget_alert_ttl: 86400 });
  });

  it("sends a String field's text rather than the change event", async () => {
    const withStringField = [
      ...PROXY_SETTINGS,
      {
        field_name: "region_name",
        field_type: "String",
        field_value: "us-east",
        field_default_value: "us-east",
        field_description: "Region to watch",
        stored_in_db: false,
        premium_field: false,
      },
    ];
    const { calls, user } = await renderPage(withStringField);

    await user.type(screen.getByDisplayValue("us-east"), "-2");
    await save(user);

    expect(alertingArgsBody(calls)).toEqual({
      daily_report_frequency: 43200,
      budget_alert_ttl: 86400,
      region_name: "us-east-2",
    });
  });

  it("keeps the edited text visible instead of replacing it with the stringified event", async () => {
    const withStringField = [
      ...PROXY_SETTINGS,
      {
        field_name: "region_name",
        field_type: "String",
        field_value: "us-east",
        field_default_value: "us-east",
        field_description: "Region to watch",
        stored_in_db: false,
        premium_field: false,
      },
    ];
    const { user } = await renderPage(withStringField);

    await user.type(screen.getByDisplayValue("us-east"), "-2");

    expect(screen.getByDisplayValue("us-east-2")).toBeInTheDocument();
  });
});

describe("AlertingSettings save feedback", () => {
  it("reports the failure and claims no success when the proxy rejects the write", async () => {
    const { user } = await renderPage(PROXY_SETTINGS, 400);

    await user.click(screen.getByRole("switch"));
    await save(user);

    expect(toastMock.fromError).toHaveBeenCalledTimes(1);
    expect(toastMock.success).not.toHaveBeenCalled();
  });

  it("confirms success when the proxy accepts the write", async () => {
    const { user } = await renderPage(PROXY_SETTINGS);

    await user.click(screen.getByRole("switch"));
    await save(user);

    expect(toastMock.success).toHaveBeenCalledWith("Wait 10s for proxy to update.");
    expect(toastMock.fromError).not.toHaveBeenCalled();
  });

  it("does not write anything when no field was changed", async () => {
    const { calls, user } = await renderPage(PROXY_SETTINGS);

    await user.click(screen.getByRole("button", { name: "Update Settings" }));

    expect(calls.filter((call) => call.url.includes("/config/field/update"))).toHaveLength(0);
  });
});
