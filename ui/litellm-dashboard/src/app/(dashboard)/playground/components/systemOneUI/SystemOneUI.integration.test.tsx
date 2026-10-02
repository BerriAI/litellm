import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render as rtlRender, screen, waitForElementToBeRemoved } from "@testing-library/react";
import type { ReactElement } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SystemOneUI from "./SystemOneUI";
import type { SystemOneResponse } from "./lib/schemas";

const responseBody: SystemOneResponse = {
  model: "jev-1.13.0",
  answers: {
    department: {
      type: "choice",
      choice: "technical",
      confidence: 0.9,
      probabilities: { technical: 0.9, billing: 0.1 },
    },
  },
};

const render = (ui: ReactElement) =>
  rtlRender(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { mutations: { retry: false } } })}>
      {ui}
    </QueryClientProvider>,
  );

const createResponse = (body: SystemOneResponse, status = 200, errorText = "") =>
  ({
    ok: status >= 200 && status < 300,
    status,
    text: async () => (status >= 200 && status < 300 ? JSON.stringify(body) : errorText),
  }) as Response;

describe("SystemOneUI integration", () => {
  const mockFetch = vi.fn<typeof fetch>();

  beforeEach(() => {
    sessionStorage.clear();
    mockFetch.mockReset();
    vi.stubGlobal("fetch", mockFetch);
    mockFetch.mockResolvedValue(createResponse(responseBody));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("restores the example after the request is edited", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    const editor = screen.getByRole("textbox", { name: "System One JSON payload" });
    const example = (editor as HTMLTextAreaElement).value;
    const resetButton = screen.getByRole("button", { name: "Reset example" });
    expect(resetButton).toBeDisabled();

    fireEvent.change(editor, { target: { value: "{}" } });
    expect(resetButton).toBeEnabled();
    await user.click(resetButton);

    expect(editor).toHaveValue(example);
    expect(resetButton).toBeDisabled();
  });

  it("flags the tab as a TypeSafe-only beta without announcing it as an alert", () => {
    render(<SystemOneUI accessToken="session-key" />);

    expect(screen.getByRole("note", { name: "System One beta notice" })).toHaveTextContent(
      "Support for more System One-compatible models is in progress.",
    );
    expect(screen.getByRole("link", { name: "Give us feedback on what you want for decision models" })).toHaveAttribute(
      "href",
      "https://github.com/BerriAI/litellm/discussions/44231",
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows invalid JSON and disables Send", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    fireEvent.change(screen.getByRole("textbox", { name: "System One JSON payload" }), {
      target: { value: "{" },
    });

    expect(await screen.findByText(/Invalid JSON syntax/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("posts the request with the session key and renders calibrated answers", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem("customProxyBaseUrl", "https://stale.example.com/");
    render(<SystemOneUI accessToken="session-key" />);
    sessionStorage.setItem("customProxyBaseUrl", "https://tenant.example.com/");

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findAllByText("technical")).toHaveLength(2);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, options] = mockFetch.mock.calls[0] ?? [];
    expect(url).toBe("https://tenant.example.com/typesafe/v1/systemone");
    expect(options?.method).toBe("POST");
    expect(options?.headers).toMatchObject({ "Content-Type": "application/json" });
    expect(Object.values(options?.headers as Record<string, string>)).toContain("Bearer session-key");
    expect(JSON.parse(options?.body as string)).toMatchObject({
      model: "jev-latest",
      questions: { has_repro_steps: { type: "noul" } },
    });
  });

  it("uses the requested model when the response omits it", async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce(
      createResponse({
        answers: responseBody.answers,
      }),
    );
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("jev-latest")).toBeInTheDocument();
    expect(screen.getByText("Selected choice")).toBeInTheDocument();
  });

  it("keeps the preview usable when question values have invalid types", () => {
    render(<SystemOneUI accessToken="session-key" />);

    fireEvent.change(screen.getByRole("textbox", { name: "System One JSON payload" }), {
      target: {
        value: JSON.stringify({
          state: "A new support request",
          questions: {
            category: {
              type: "choice",
              instructions: { text: "Route the message" },
              criteria: { support: { text: "Help" } },
            },
          },
        }),
      },
    });

    expect(screen.getByText("Instructions must be a string.")).toBeInTheDocument();
    expect(screen.getByText("Choice descriptions must be strings.")).toBeInTheDocument();
    expect(screen.getByText("Enter a valid request to preview its state and questions.")).toBeInTheDocument();
  });

  it("renders upstream errors inline", async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce(createResponse(responseBody, 401, "Virtual key rejected"));
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Virtual key rejected");
  });

  it("renders malformed success responses inline", async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () =>
        JSON.stringify({
          model: "jev-1.13.0",
          answers: {
            department: {
              type: "choice",
              choice: "technical",
              probabilities: null,
            },
          },
        }),
    } as Response);
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("System One response has an invalid shape.");
    expect(screen.queryByText("jev-1.13.0")).not.toBeInTheDocument();
  });

  it("clears a previous answer when the request editor changes", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "System One JSON payload" }), {
      target: {
        value: JSON.stringify({
          state: "A different support request",
          questions: {
            department: {
              type: "choice",
              instructions: "Where should this go?",
              criteria: { billing: "Billing questions" },
            },
          },
        }),
      },
    });

    expect(screen.queryByText("Selected choice")).not.toBeInTheDocument();
    expect(screen.queryByText("jev-1.13.0")).not.toBeInTheDocument();
  });

  it("clears the answer when the UI session key changes", async () => {
    const user = userEvent.setup();
    const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    const view = rtlRender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="session-key" />
      </QueryClientProvider>,
    );

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();

    view.rerender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="new-session-key" />
      </QueryClientProvider>,
    );

    await waitForElementToBeRemoved(() => screen.queryByText("Selected choice"));
  });

  it("aborts an in-flight request when cancelled", async () => {
    const user = userEvent.setup();
    mockFetch.mockImplementation(
      (_input, init) =>
        new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("Request aborted", "AbortError")), {
            once: true,
          });
        }),
    );
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));
    await user.click(await screen.findByRole("button", { name: "Cancel request" }));

    expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
  });
});
