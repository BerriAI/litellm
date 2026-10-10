import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render as rtlRender,
  screen,
  waitFor,
  waitForElementToBeRemoved,
} from "@testing-library/react";
import type { ReactElement } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SystemOneUI from "./SystemOneUI";
import type { OpenAIDecisionsResponse, SystemOneResponse } from "./lib/schemas";

interface ModelGroupInfo {
  model_group: string;
  mode: string;
}

const DECISION_AND_CHAT_MODELS: ModelGroupInfo[] = [
  { model_group: "gpt-5.5", mode: "chat" },
  { model_group: "jev-latest", mode: "evaluation" },
  { model_group: "pplx-decider", mode: "evaluation" },
];

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

const openAIResponseBody: OpenAIDecisionsResponse = {
  model: "jev-1.13.0",
  answers: [
    {
      type: "choice",
      name: "area",
      choice: "backend",
      confidence: 0.9,
      probabilities: [
        { value: "backend", probability: 0.9 },
        { value: "sdk", probability: 0.1 },
      ],
    },
    { type: "predicate", name: null, probability: 0.25 },
    {
      type: "score",
      name: "severity",
      score: 2.6,
      confidence: 0.5,
      probabilities: [
        { value: 0, label: "Cosmetic", probability: 0.1 },
        { value: 3, label: "Broken", probability: 0.9 },
      ],
    },
  ],
  usage: { input_tokens: 10, output_tokens: 2 },
};

const render = (ui: ReactElement) =>
  rtlRender(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { mutations: { retry: false } } })}>
      {ui}
    </QueryClientProvider>,
  );

const modelGroupInfoResponse = (groups: ModelGroupInfo[]) =>
  ({ ok: true, status: 200, text: async () => JSON.stringify({ data: groups }) }) as Response;

const EXPIRED_KEY_RESPONSE = {
  ok: false,
  status: 401,
  text: async () =>
    JSON.stringify({ error: { message: "Authentication Error - Expired Key.", type: "expired_key", code: "401" } }),
} as Response;

const isModelLookup = (input: RequestInfo | URL) => String(input).endsWith("/model_group/info");

const lookupAuthHeaders = (calls: Parameters<typeof fetch>[]) =>
  calls.map(([, init]) => Object.values((init?.headers ?? {}) as Record<string, string>));

const createResponse = (body: SystemOneResponse | OpenAIDecisionsResponse, status = 200, errorText = "") =>
  ({
    ok: status >= 200 && status < 300,
    status,
    text: async () => (status >= 200 && status < 300 ? JSON.stringify(body) : errorText),
  }) as Response;

describe("SystemOneUI integration", () => {
  const mockFetch = vi.fn<typeof fetch>();
  const mockModelLookup = vi.fn<typeof fetch>();

  beforeEach(() => {
    sessionStorage.clear();
    mockFetch.mockReset();
    mockModelLookup.mockReset();
    vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) =>
      isModelLookup(input) ? mockModelLookup(input, init) : mockFetch(input, init),
    );
    mockFetch.mockResolvedValue(createResponse(responseBody));
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse([]));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
    document.cookie = "token=; path=/; max-age=0";
  });

  it("restores the example after the request is edited", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    const editor = screen.getByRole("textbox", { name: "Decisions JSON payload" });
    const example = (editor as HTMLTextAreaElement).value;
    const resetButton = screen.getByRole("button", { name: "Reset example" });
    expect(resetButton).toBeDisabled();

    fireEvent.change(editor, { target: { value: "{}" } });
    expect(resetButton).toBeEnabled();
    await user.click(resetButton);

    expect(editor).toHaveValue(example);
    expect(resetButton).toBeDisabled();
  });

  it("explains the native decision endpoint after opting in without announcing it as an alert", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "System One · /v1/systemone" }));

    expect(screen.getByRole("note", { name: "Decision endpoint notice" })).toHaveTextContent(
      "omit model to use the proxy's configured default.",
    );
    expect(screen.getByRole("link", { name: "How to call /v1/decisions and /v1/systemone" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/decisions",
    );
    expect(screen.getByRole("link", { name: "Give us feedback on what you want for decision models" })).toHaveAttribute(
      "href",
      "https://github.com/BerriAI/litellm/discussions/44231",
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("opens on /v1/decisions with the first decision model and renders OpenAI-shaped answers", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    mockFetch.mockResolvedValue(createResponse(openAIResponseBody));
    render(<SystemOneUI accessToken="session-key" />);

    expect(await screen.findByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest");
    expect(lookupAuthHeaders(mockModelLookup.mock.calls)).toEqual([expect.arrayContaining(["Bearer session-key"])]);
    expect(screen.getByRole("heading", { name: "Input" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "has_repro_steps" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/decisions$/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toMatchObject({
      model: "jev-latest",
      input: expect.any(String),
      questions: [{ type: "choice", name: "area" }, { type: "predicate" }, { type: "score", name: "severity" }],
    });
    expect(screen.getByText("q1")).toBeInTheDocument();
    expect(screen.getByText("25% yes")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "3: Broken probability" })).toHaveAttribute(
      "aria-valuetext",
      "90%, selected",
    );
    expect(screen.getByText("10 input / 2 output tokens")).toBeInTheDocument();
  });

  it("switches to the System One shape on /v1/systemone", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);
    await screen.findByRole("combobox", { name: "Decision model" });
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "System One · /v1/systemone" }));

    expect(screen.getByRole("heading", { name: "State" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(mockFetch.mock.calls[0]?.[0]).not.toMatch(/typesafe/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toMatchObject({
      model: "jev-latest",
      questions: { has_repro_steps: { type: "noul" } },
    });
  });

  it("offers only decision models and writes the picked one into the request", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);
    const picker = await screen.findByRole("combobox", { name: "Decision model" });

    await user.clear(picker);
    await user.type(picker, "p");
    expect(await screen.findByRole("option", { name: "pplx-decider" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "gpt-5.5" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: "pplx-decider" }));

    const editor = screen.getByRole("textbox", { name: "Decisions JSON payload" });
    expect(JSON.parse((editor as HTMLTextAreaElement).value)).toMatchObject({
      model: "pplx-decider",
      questions: [{ type: "choice" }, { type: "predicate" }, { type: "score" }],
    });
    mockFetch.mockResolvedValue(createResponse(openAIResponseBody));
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string).model).toBe("pplx-decider");
  });

  it("links to Add Model when the key cannot call any decision model", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "System One · /v1/systemone" }));

    expect(await screen.findByRole("link", { name: "Add a decision model" })).toHaveAttribute(
      "href",
      "/ui/models-and-endpoints",
    );
    expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveAttribute(
      "placeholder",
      "No decision models yet",
    );
  });

  it("looks up a typed virtual key's decision models once, after typing settles", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken={null} disabledPersonalKeyCreation />);

    await user.type(screen.getByLabelText("Virtual Key", { exact: true }), "sk-typed-key");

    expect(await screen.findByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest");
    expect(lookupAuthHeaders(mockModelLookup.mock.calls)).toEqual([expect.arrayContaining(["Bearer sk-typed-key"])]);
  });

  it("keeps the admin signed in when a typed virtual key has expired", async () => {
    document.cookie = "token=ui-session; path=/";
    mockModelLookup.mockResolvedValue(EXPIRED_KEY_RESPONSE);
    render(<SystemOneUI accessToken={null} disabledPersonalKeyCreation />);

    fireEvent.change(screen.getByLabelText("Virtual Key", { exact: true }), { target: { value: "sk-expired" } });

    await waitFor(() => expect(mockModelLookup).toHaveBeenCalled());
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(document.cookie).toContain("token=ui-session");
    expect(screen.getByRole("combobox", { name: "Decision endpoint" })).toHaveTextContent("/typesafe/v1/systemone");
  });

  it("keeps an edited TypeSafe request in place when decision models load afterwards", async () => {
    const models = Promise.withResolvers<Response>();
    mockModelLookup.mockReturnValue(models.promise);
    render(<SystemOneUI accessToken="session-key" />);
    const editor = screen.getByRole("textbox", { name: "Decisions JSON payload" });
    const draft = JSON.stringify({ state: "edited", questions: { q: { type: "noul", instructions: "Yes?" } } });
    fireEvent.change(editor, { target: { value: draft } });

    await act(async () => {
      models.resolve(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(editor).toHaveValue(draft);
    expect(screen.queryByRole("combobox", { name: "Decision model" })).not.toBeInTheDocument();
  });

  it("shows invalid JSON and disables Send", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    fireEvent.change(screen.getByRole("textbox", { name: "Decisions JSON payload" }), {
      target: { value: "{" },
    });

    expect(await screen.findByText(/Invalid JSON syntax/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("preserves the untouched TypeSafe default and renders calibrated answers", async () => {
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

  it("keeps legacy validation when question values have invalid types", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    fireEvent.change(screen.getByRole("textbox", { name: "Decisions JSON payload" }), {
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
    expect(screen.getByText("Enter a valid request to preview its input and questions.")).toBeInTheDocument();
  });

  it("preserves each endpoint draft and sends legacy requests to TypeSafe", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "System One · /v1/systemone" }));
    const editor = screen.getByRole("textbox", { name: "Decisions JSON payload" });
    const draft = JSON.stringify({
      model: "my-decider",
      state: {},
      questions: {
        route: { type: "choice", criteria: { support: { text: "Help" }, other: null } },
      },
    });
    fireEvent.change(editor, { target: { value: draft } });
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
    expect(screen.getByText(/"text": "Help"/)).toBeInTheDocument();

    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "TypeSafe · /typesafe/v1/systemone" }));
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/typesafe\/v1\/systemone$/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string).model).toBe("jev-latest");

    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "System One · /v1/systemone" }));
    expect(editor).toHaveValue(draft);
    expect(screen.queryByText("Selected choice")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[1]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(JSON.parse(mockFetch.mock.calls[1]?.[1]?.body as string)).toEqual(JSON.parse(draft));
  });

  it("sends a native request without model so the proxy can select its default", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "System One · /v1/systemone" }));
    const payload = { state: "An outage", questions: { urgent: { type: "noul", instructions: "Is this urgent?" } } };
    fireEvent.change(screen.getByRole("textbox", { name: "Decisions JSON payload" }), {
      target: { value: JSON.stringify(payload) },
    });
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("jev-1.13.0")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    const body = JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string);
    expect(body).toEqual(payload);
    expect(body).not.toHaveProperty("model");
  });

  it.each(["success", "error"])("ignores a late %s after switching endpoints", async (outcome) => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Response>();
    mockFetch.mockReturnValueOnce(pending.promise);
    const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    rtlRender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="session-key" />
      </QueryClientProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByRole("button", { name: "Cancel request" });
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "System One · /v1/systemone" }));
    expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();

    await act(async () => {
      if (outcome === "success") {
        pending.resolve(createResponse({ model: "stale-model", answers: { stale: { type: "noul", noul: 1 } } }));
      } else {
        pending.reject(new Error("Stale request failed"));
      }
      await pending.promise.catch(() => undefined);
    });
    await waitFor(() => expect(queryClient.isMutating()).toBe(0));
    expect(screen.queryByText("stale-model")).not.toBeInTheDocument();
    expect(screen.queryByText("Stale request failed")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText("100% yes")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("jev-1.13.0")).toBeInTheDocument();
    expect(screen.getByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[1]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("uses a custom virtual key when personal key creation is disabled", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken={null} disabledPersonalKeyCreation />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Virtual Key", { exact: true }), {
      target: { value: "test-virtual-key" },
    });
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(Object.values(mockFetch.mock.calls[0]?.[1]?.headers ?? {})).toContain("Bearer test-virtual-key");
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

    expect(await screen.findByRole("alert")).toHaveTextContent("Decisions response has an invalid shape.");
    expect(screen.queryByText("jev-1.13.0")).not.toBeInTheDocument();
  });

  it("clears a previous answer when the request editor changes", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Decisions JSON payload" }), {
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
