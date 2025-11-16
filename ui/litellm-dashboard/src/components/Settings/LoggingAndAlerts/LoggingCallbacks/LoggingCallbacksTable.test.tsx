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
});
