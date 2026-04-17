import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ContentRoutingConfiguration from "./ContentRoutingConfiguration";
import type { ContentRoutingConfig } from "./ContentRoutingConfiguration";

const defaultConfig: ContentRoutingConfig = {
  enabled: false,
  classifier: "rule_based",
};

const enabledConfig: ContentRoutingConfig = {
  enabled: true,
  classifier: "rule_based",
  default_model: "",
  confidence_threshold: 0.1,
};

const baseProps = {
  config: defaultConfig,
  onChange: vi.fn(),
  accessToken: "test-token",
};

describe("ContentRoutingConfiguration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ── Rendering ────────────────────────────────────────────────────────────────

  it("renders the section heading", () => {
    render(<ContentRoutingConfiguration {...baseProps} />);
    expect(screen.getByText("Content-Aware Routing")).toBeInTheDocument();
  });

  it("renders the enable toggle", () => {
    render(<ContentRoutingConfiguration {...baseProps} />);
    expect(screen.getByRole("switch")).toBeInTheDocument();
  });

  it("does not render classifier options when disabled", () => {
    render(<ContentRoutingConfiguration {...baseProps} />);
    expect(screen.queryByText("Classifier")).not.toBeInTheDocument();
  });

  it("renders classifier and common fields when enabled", () => {
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );
    expect(screen.getByText("Classifier")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g. gpt-4o")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("0.1")).toBeInTheDocument();
  });

  // ── Toggle ───────────────────────────────────────────────────────────────────

  it("calls onChange with enabled=true when toggle is clicked", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration
        {...baseProps}
        onChange={onChange}
        config={defaultConfig}
      />
    );

    await user.click(screen.getByRole("switch"));

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: true })
    );
  });

  it("calls onChange with enabled=false when toggled off", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration
        {...baseProps}
        onChange={onChange}
        config={enabledConfig}
      />
    );

    await user.click(screen.getByRole("switch"));

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: false })
    );
  });

  // ── Classifier descriptions ───────────────────────────────────────────────────

  it("shows rule_based description by default", () => {
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );
    expect(screen.getByText(/TF-IDF keyword matching/i)).toBeInTheDocument();
  });

  // ── Conditional fields: embedding_similarity ──────────────────────────────────

  it("shows embedding model field when classifier is embedding_similarity", () => {
    render(
      <ContentRoutingConfiguration
        {...baseProps}
        config={{ ...enabledConfig, classifier: "embedding_similarity" }}
      />
    );
    expect(screen.getByText(/embedding model/i)).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("text-embedding-3-small")
    ).toBeInTheDocument();
  });

  it("does not show embedding model field for rule_based classifier", () => {
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );
    expect(screen.queryByText(/embedding model/i)).not.toBeInTheDocument();
  });

  // ── Conditional fields: external_model ────────────────────────────────────────

  it("shows external classifier URL field when classifier is external_model", () => {
    render(
      <ContentRoutingConfiguration
        {...baseProps}
        config={{ ...enabledConfig, classifier: "external_model" }}
      />
    );
    expect(screen.getByText(/external classifier url/i)).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/arch-router/i)
    ).toBeInTheDocument();
  });

  it("does not show external classifier URL for rule_based classifier", () => {
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );
    expect(
      screen.queryByText(/external classifier url/i)
    ).not.toBeInTheDocument();
  });

  // ── Default model field ───────────────────────────────────────────────────────

  it("calls onChange with updated default_model when edited", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration
        {...baseProps}
        onChange={onChange}
        config={enabledConfig}
      />
    );

    await user.type(screen.getByPlaceholderText("e.g. gpt-4o"), "g");

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ default_model: "g" })
    );
  });

  // ── Confidence threshold field ────────────────────────────────────────────────

  it("renders confidence threshold with the configured value", () => {
    render(
      <ContentRoutingConfiguration
        {...baseProps}
        config={{ ...enabledConfig, confidence_threshold: 0.75 }}
      />
    );
    expect(screen.getByDisplayValue("0.75")).toBeInTheDocument();
  });

  it("calls onChange with updated confidence_threshold when valid number is entered", () => {
    const onChange = vi.fn();
    render(
      <ContentRoutingConfiguration
        {...baseProps}
        onChange={onChange}
        config={{ ...enabledConfig, confidence_threshold: 0.1 }}
      />
    );

    const input = screen.getByDisplayValue("0.1");
    fireEvent.change(input, { target: { value: "0.5" } });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ confidence_threshold: 0.5 })
    );
  });

  // ── Test panel ────────────────────────────────────────────────────────────────

  it("shows 'Open tester' button when enabled", () => {
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );
    expect(
      screen.getByRole("button", { name: /open tester/i })
    ).toBeInTheDocument();
  });

  it("toggles the test panel visibility when 'Open tester' is clicked", async () => {
    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    const toggleBtn = screen.getByRole("button", { name: /open tester/i });
    await user.click(toggleBtn);

    expect(
      screen.getByPlaceholderText(/write a python function/i)
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /hide/i })
    ).toBeInTheDocument();
  });

  it("hides the test panel again when 'Hide' is clicked", async () => {
    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.click(screen.getByRole("button", { name: /hide/i }));

    expect(
      screen.queryByPlaceholderText(/write a python function/i)
    ).not.toBeInTheDocument();
  });

  it("disables 'Run classification' when the prompt is empty", async () => {
    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));

    expect(
      screen.getByRole("button", { name: /run classification/i })
    ).toBeDisabled();
  });

  it("enables 'Run classification' after the user types a prompt", async () => {
    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "hello"
    );

    expect(
      screen.getByRole("button", { name: /run classification/i })
    ).not.toBeDisabled();
  });

  // ── runTest — successful response ─────────────────────────────────────────────

  it("displays classification results after a successful API call", async () => {
    const mockResult = {
      matched_preference: "coding",
      matched_model: "gpt-4o",
      confidence: 0.9123,
      classifier: "rule_based",
    };
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    } as Response);

    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "sort a list"
    );
    await user.click(screen.getByRole("button", { name: /run classification/i }));

    await waitFor(() =>
      expect(screen.getByText("coding")).toBeInTheDocument()
    );
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("0.9123")).toBeInTheDocument();
  });

  it("posts to /utils/content_route_test with the correct payload", async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        matched_preference: "coding",
        matched_model: "gpt-4o",
        confidence: 0.9,
        classifier: "rule_based",
      }),
    } as Response);
    global.fetch = mockFetch;

    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "my prompt"
    );
    await user.click(screen.getByRole("button", { name: /run classification/i }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledOnce());

    const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/utils/content_route_test");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ prompt: "my prompt" });
    expect((init.headers as Record<string, string>)["Authorization"]).toBe(
      "Bearer test-token"
    );
  });

  // ── runTest — error response ───────────────────────────────────────────────────

  it("displays an error message when the API call fails", async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: "Router not configured" }),
    } as Response);

    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "something"
    );
    await user.click(screen.getByRole("button", { name: /run classification/i }));

    await waitFor(() =>
      expect(screen.getByText("Router not configured")).toBeInTheDocument()
    );
  });

  it("displays a fallback error message when response has no detail field", async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({}),
    } as Response);

    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "something"
    );
    await user.click(screen.getByRole("button", { name: /run classification/i }));

    await waitFor(() =>
      expect(screen.getByText(/HTTP 422/i)).toBeInTheDocument()
    );
  });

  it("displays an error message when fetch throws a network error", async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new Error("Network failure"));

    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "something"
    );
    await user.click(screen.getByRole("button", { name: /run classification/i }));

    await waitFor(() =>
      expect(screen.getByText("Network failure")).toBeInTheDocument()
    );
  });

  // ── all_scores details block ──────────────────────────────────────────────────

  it("renders all_scores when present in the result", async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        matched_preference: "coding",
        matched_model: "gpt-4o",
        confidence: 0.9,
        classifier: "rule_based",
        all_scores: { coding: 0.9, writing: 0.1 },
      }),
    } as Response);

    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration {...baseProps} config={enabledConfig} />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "sort a list"
    );
    await user.click(screen.getByRole("button", { name: /run classification/i }));

    await waitFor(() =>
      expect(screen.getByText(/all scores/i)).toBeInTheDocument()
    );
  });

  // ── accessToken guard ─────────────────────────────────────────────────────────

  it("does not fetch when accessToken is null", async () => {
    const mockFetch = vi.fn();
    global.fetch = mockFetch;

    const user = userEvent.setup();
    render(
      <ContentRoutingConfiguration
        config={enabledConfig}
        onChange={vi.fn()}
        accessToken={null}
      />
    );

    await user.click(screen.getByRole("button", { name: /open tester/i }));
    await user.type(
      screen.getByPlaceholderText(/write a python function/i),
      "hello"
    );
    await user.click(screen.getByRole("button", { name: /run classification/i }));

    expect(mockFetch).not.toHaveBeenCalled();
  });
});
