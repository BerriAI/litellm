import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { toast } from "@/lib/toast";
import { testConnectionRequest } from "../networking";
import { prepareModelAddRequest } from "./handle_add_model_submit";
import ModelConnectionTest from "./model_connection_test";

vi.mock("../networking", () => ({ testConnectionRequest: vi.fn() }));
vi.mock("./handle_add_model_submit", () => ({ prepareModelAddRequest: vi.fn() }));

const preparedRequest = [{ litellmParamsObj: { model: "openai/gpt-4o-mini" }, modelInfoObj: { mode: "chat" } }];

const finishConnectionTest = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300);
  });
};

describe("ModelConnectionTest", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
    vi.mocked(prepareModelAddRequest).mockResolvedValue(preparedRequest as never);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("tests the prepared model and shows a successful result", async () => {
    const onTestComplete = vi.fn();
    vi.mocked(testConnectionRequest).mockResolvedValue({ status: "success" } as never);

    render(
      <ModelConnectionTest
        formValues={{ model: "gpt-4o-mini" }}
        accessToken="sk-test"
        testMode="chat"
        modelName="GPT-4o mini"
        onTestComplete={onTestComplete}
      />,
    );

    expect(screen.getByText("Testing connection to GPT-4o mini...")).toBeInTheDocument();
    await finishConnectionTest();

    expect(prepareModelAddRequest).toHaveBeenCalledWith({ model: "gpt-4o-mini" }, "sk-test", null);
    expect(testConnectionRequest).toHaveBeenCalledWith(
      "sk-test",
      { model: "openai/gpt-4o-mini" },
      { mode: "chat" },
      "chat",
    );
    expect(screen.getByTestId("connection-success-msg")).toHaveTextContent("Connection to GPT-4o mini successful!");
    expect(toast.success).toHaveBeenCalledWith("Connection test successful!");
    expect(onTestComplete).toHaveBeenCalledTimes(1);
  });

  it("shows a cleaned provider error, request details, and copies the curl command", async () => {
    const writeText = vi.fn();
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    vi.mocked(testConnectionRequest).mockResolvedValue({
      status: "error",
      result: {
        error: "litellm.AuthenticationError: invalid api key stack trace: hidden",
        raw_request_typed_dict: {
          raw_request_api_base: "https://api.example.test/v1/chat/completions",
          raw_request_body: { model: "gpt-4o-mini" },
          raw_request_headers: { Authorization: "Bearer test" },
        },
      },
    } as never);

    render(
      <ModelConnectionTest
        formValues={{ model: "gpt-4o-mini" }}
        accessToken="sk-test"
        testMode="chat"
        modelName="GPT-4o mini"
      />,
    );
    await finishConnectionTest();

    expect(screen.getByTestId("connection-failure-msg")).toHaveTextContent("Connection to GPT-4o mini failed");
    expect(screen.getByText("invalid api key")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show Details" }));
    expect(screen.getByText("Troubleshooting Details")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Copy to Clipboard/ }));

    expect(writeText).toHaveBeenCalledWith(expect.stringContaining("https://api.example.test/v1/chat/completions"));
    expect(toast.success).toHaveBeenCalledWith("Copied to clipboard");
  });

  it("shows a preparation failure without sending a connection request", async () => {
    vi.mocked(prepareModelAddRequest).mockResolvedValue(null as never);

    render(<ModelConnectionTest formValues={{}} accessToken="sk-test" testMode="chat" />);
    await finishConnectionTest();

    expect(screen.getByText("Failed to prepare model data. Please check your form inputs.")).toBeInTheDocument();
    expect(testConnectionRequest).not.toHaveBeenCalled();
  });
});
