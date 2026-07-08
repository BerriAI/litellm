import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
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
});
