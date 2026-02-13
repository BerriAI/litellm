import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { AlertingObject, AlertingVariables } from "./types";
import { LoggingCallbacksTable } from "./LoggingCallbacksTable";

const emptyVariables = {} as AlertingVariables;

const defaultTableVariables: AlertingVariables = {
  SLACK_WEBHOOK_URL: null,
  LANGFUSE_PUBLIC_KEY: null,
  LANGFUSE_SECRET_KEY: null,
  LANGFUSE_HOST: null,
  OPENMETER_API_KEY: null,
};

const langfuseCallback = (overrides?: Partial<AlertingObject>): AlertingObject => ({
  name: "langfuse",
  type: "success",
  variables: emptyVariables,
  ...overrides,
});

const defaultProps = {
  callbacks: [] as AlertingObject[],
  availableCallbacks: {} as Record<string, { litellm_callback_name: string; litellm_callback_params: string[]; ui_callback_name: string }>,
};

describe("LoggingCallbacksTable", () => {
  it("should render", () => {
    render(<LoggingCallbacksTable {...defaultProps} />);
    expect(screen.getByText("Active Logging Callbacks")).toBeInTheDocument();
  });

  it("should show empty state when callbacks is empty", () => {
    render(<LoggingCallbacksTable {...defaultProps} />);
    expect(screen.getByText("No callbacks configured")).toBeInTheDocument();
    expect(screen.getByText(/Add your first callback/)).toBeInTheDocument();
  });

  describe("actions", () => {
    it("should call onAdd when '+ Add Callback' is clicked", async () => {
      const onAdd = vi.fn();
      const user = userEvent.setup();
      render(<LoggingCallbacksTable {...defaultProps} onAdd={onAdd} />);
      await act(async () => {
        await user.click(screen.getByRole("button", { name: /add callback/i }));
      });
      expect(onAdd).toHaveBeenCalledTimes(1);
    });

    it("should call onEdit when Edit action is clicked", async () => {
      const onEdit = vi.fn();
      const user = userEvent.setup();
      const callback = langfuseCallback({ variables: defaultTableVariables });
      render(
        <LoggingCallbacksTable {...defaultProps} callbacks={[callback]} onEdit={onEdit} />
      );
      await act(async () => {
        await user.click(screen.getByTestId("logging-callback-edit-langfuse"));
      });
      expect(onEdit).toHaveBeenCalledTimes(1);
      expect(onEdit).toHaveBeenCalledWith(callback);
    });

    it("should call onDelete when Delete action is clicked", async () => {
      const onDelete = vi.fn();
      const user = userEvent.setup();
      const callback = langfuseCallback();
      render(
        <LoggingCallbacksTable {...defaultProps} callbacks={[callback]} onDelete={onDelete} />
      );
      await act(async () => {
        await user.click(screen.getByTestId("logging-callback-delete-langfuse"));
      });
      expect(onDelete).toHaveBeenCalledTimes(1);
      expect(onDelete).toHaveBeenCalledWith(callback);
    });

    it("should call onTest when Test action is clicked", async () => {
      const onTest = vi.fn();
      const user = userEvent.setup();
      const callback = langfuseCallback();
      render(
        <LoggingCallbacksTable {...defaultProps} callbacks={[callback]} onTest={onTest} />
      );
      await act(async () => {
        await user.click(screen.getByTestId("logging-callback-test-langfuse"));
      });
      expect(onTest).toHaveBeenCalledTimes(1);
      expect(onTest).toHaveBeenCalledWith(callback);
    });
  });

  describe("mode badges", () => {
    it('should show mode badge "Success" for type success', () => {
      render(
        <LoggingCallbacksTable {...defaultProps} callbacks={[langfuseCallback()]} />
      );
      expect(screen.getByText("Success")).toBeInTheDocument();
    });

    it('should show mode badge "Failure" for type failure', () => {
      render(
        <LoggingCallbacksTable
          {...defaultProps}
          callbacks={[langfuseCallback({ name: "sentry", type: "failure" })]}
        />
      );
      expect(screen.getByText("Failure")).toBeInTheDocument();
    });
  });

  describe("display name", () => {
    it('should map "otel" to "OpenTelemetry" on the table', () => {
      render(
        <LoggingCallbacksTable
          {...defaultProps}
          callbacks={[{ name: "otel", variables: defaultTableVariables }]}
          availableCallbacks={{
            otel: {
              litellm_callback_name: "otel",
              litellm_callback_params: [],
              ui_callback_name: "OpenTelemetry",
            },
          }}
        />
      );
      expect(screen.getByText("OpenTelemetry")).toBeInTheDocument();
    });

    it("should fallback to original callback name when not in availableCallbacks", () => {
      render(
        <LoggingCallbacksTable
          {...defaultProps}
          callbacks={[{ name: "custom_callback_x", variables: defaultTableVariables }]}
        />
      );
      expect(screen.getByText("custom_callback_x")).toBeInTheDocument();
    });

    it("should display Success & Failure when callback has type success_and_failure", () => {
      render(
        <LoggingCallbacksTable
          {...defaultProps}
          callbacks={[
            {
              name: "websearch_interception",
              type: "success_and_failure",
              variables: defaultTableVariables,
              params: { enabled_providers: ["bedrock"] },
            },
          ]}
          availableCallbacks={{
            websearch_interception: {
              litellm_callback_name: "websearch_interception",
              litellm_callback_params: ["enabled_providers", "search_tool_name"],
              ui_callback_name: "WebSearch Interception",
            },
          }}
        />
      );
      expect(screen.getByText("WebSearch Interception")).toBeInTheDocument();
      expect(screen.getByText("Success & Failure")).toBeInTheDocument();
    });
  });
});
