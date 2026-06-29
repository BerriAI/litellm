import { fireEvent, render, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LoggingCallbacksTable } from "./LoggingCallbacksTable";

describe("LoggingCallbacksTable", () => {
  it("should render", () => {
    const { getByText } = render(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} />);
    expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
  });

  it('should map "otel" to "OpenTelemetry" on the table', () => {
    const { getByText } = render(
      <LoggingCallbacksTable
        callbacks={[
          {
            name: "otel",
            variables: {
              SLACK_WEBHOOK_URL: null,
              LANGFUSE_PUBLIC_KEY: null,
              LANGFUSE_SECRET_KEY: null,
              LANGFUSE_HOST: null,
              OPENMETER_API_KEY: null,
            },
          },
        ]}
        availableCallbacks={{
          otel: {
            litellm_callback_name: "otel",
            litellm_callback_params: [],
            ui_callback_name: "OpenTelemetry",
          },
        }}
      />,
    );
    expect(getByText("OpenTelemetry")).toBeInTheDocument();
  });

  it("should fallback to original callback name when not in availableCallbacks", () => {
    const { getByText } = render(
      <LoggingCallbacksTable
        callbacks={[
          {
            name: "custom_callback_x",
            variables: {
              SLACK_WEBHOOK_URL: null,
              LANGFUSE_PUBLIC_KEY: null,
              LANGFUSE_SECRET_KEY: null,
              LANGFUSE_HOST: null,
              OPENMETER_API_KEY: null,
            },
          },
        ]}
        availableCallbacks={{}}
      />,
    );
    expect(getByText("custom_callback_x")).toBeInTheDocument();
  });

  // Regression: `/get_callbacks` returns the same `name` twice when a
  // callback is registered for both success and failure (e.g. `generic_api`
  // → POST to spend-log on both 200 and 4xx/5xx). The UI used to ignore
  // the `type` field and render every row as "Success", masking the
  // failure registration. Reading `record.type` fixes the badge AND
  // composing the rowKey with type avoids React's duplicate-key warning.
  it("renders distinct Success and Failure badges for same-name dual registration", () => {
    const baseVars = {
      SLACK_WEBHOOK_URL: null,
      LANGFUSE_PUBLIC_KEY: null,
      LANGFUSE_SECRET_KEY: null,
      LANGFUSE_HOST: null,
      OPENMETER_API_KEY: null,
    };
    const { getAllByText, getByText } = render(
      <LoggingCallbacksTable
        callbacks={[
          { name: "generic_api", type: "success", variables: baseVars },
          { name: "generic_api", type: "failure", variables: baseVars },
        ]}
        availableCallbacks={{
          generic_api: {
            litellm_callback_name: "generic_api",
            litellm_callback_params: [],
            ui_callback_name: "Custom Callback API",
          },
        }}
      />,
    );
    // Both rows show the same display name, but distinct mode badges.
    expect(getAllByText("Custom Callback API")).toHaveLength(2);
    expect(getByText("Success")).toBeInTheDocument();
    expect(getByText("Failure")).toBeInTheDocument();
  });

  const NO_VARS = {
    SLACK_WEBHOOK_URL: null,
    LANGFUSE_PUBLIC_KEY: null,
    LANGFUSE_SECRET_KEY: null,
    LANGFUSE_HOST: null,
    OPENMETER_API_KEY: null,
  };

  it("renders a global destination's scope as a Global tag with no mode badge", () => {
    const { getByText, queryByText } = render(
      <LoggingCallbacksTable
        callbacks={[
          {
            name: "langfuse-eu",
            variables: NO_VARS,
            credentialName: "langfuse-eu",
            access: { global: true },
            resolvedScope: { global: true, teams: [], orgs: [] },
          },
        ]}
        availableCallbacks={{}}
      />,
    );
    expect(getByText("Global")).toBeInTheDocument();
    // a destination has no success/failure mode badge
    expect(queryByText("Success")).not.toBeInTheDocument();
  });

  it("renders a scoped destination's resolved teams and orgs as labeled tags", () => {
    const { getByText } = render(
      <LoggingCallbacksTable
        callbacks={[
          {
            name: "arize-eu",
            variables: NO_VARS,
            credentialName: "arize-eu",
            access: { teams: ["t1", "t2"], orgs: ["o1"] },
            resolvedScope: { global: false, teams: ["t1", "t2"], orgs: ["o1"] },
          },
        ]}
        availableCallbacks={{}}
      />,
    );
    expect(getByText("team: t1")).toBeInTheDocument();
    expect(getByText("team: t2")).toBeInTheDocument();
    expect(getByText("org: o1")).toBeInTheDocument();
  });

  it("a destination row fires onEditAccess and onDelete, never onTest", () => {
    const onEditAccess = vi.fn();
    const onDelete = vi.fn();
    const onTest = vi.fn();
    const { getByText } = render(
      <LoggingCallbacksTable
        callbacks={[{ name: "dest", variables: NO_VARS, credentialName: "dest", access: { global: true } }]}
        availableCallbacks={{}}
        onEditAccess={onEditAccess}
        onDelete={onDelete}
        onTest={onTest}
      />,
    );
    const row = getByText("dest").closest("tr") as HTMLElement;
    const scoped = within(row);
    // destination rows expose edit-access + delete, and no test action
    expect(scoped.queryByTestId("test-callback")).not.toBeInTheDocument();
    fireEvent.click(scoped.getByTestId("edit-access"));
    fireEvent.click(scoped.getByTestId("delete-destination"));
    expect(onEditAccess).toHaveBeenCalledWith(expect.objectContaining({ credentialName: "dest" }));
    expect(onDelete).toHaveBeenCalledWith(expect.objectContaining({ credentialName: "dest" }));
    expect(onTest).not.toHaveBeenCalled();
  });

  it("a config callback row keeps the test/edit/delete actions and an em-dash access", () => {
    const { getByText } = render(
      <LoggingCallbacksTable
        callbacks={[{ name: "datadog", type: "success", variables: NO_VARS }]}
        availableCallbacks={{}}
      />,
    );
    const row = getByText("datadog").closest("tr") as HTMLElement;
    const scoped = within(row);
    expect(scoped.getByTestId("test-callback")).toBeInTheDocument();
    expect(scoped.getByTestId("edit-callback")).toBeInTheDocument();
    expect(scoped.getByTestId("delete-callback")).toBeInTheDocument();
    expect(scoped.queryByTestId("edit-access")).not.toBeInTheDocument();
    // access cell renders an em-dash for non-destination rows
    expect(scoped.getByText("—")).toBeInTheDocument();
  });
});
