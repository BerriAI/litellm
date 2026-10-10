import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render as rtlRender,
  screen,
  waitFor,
  waitForElementToBeRemoved,
  within,
} from "@testing-library/react";
import type { ReactElement } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SystemOneUI from "./SystemOneUI";
import type { SystemOneResponse } from "./lib/schemas";
import type { OpenAIDecisionsResponse } from "./lib/openAIDecisions";

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

const openJsonEditor = async () => {
  await userEvent.click(screen.getByRole("tab", { name: "JSON" }));
  return screen.getByRole("textbox", { name: "System One JSON payload" });
};

const selectEndpoint = async (endpoint: "/v1/decisions" | "/v1/systemone") => {
  screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
  await userEvent.keyboard("{ArrowDown}");
  await userEvent.click(await screen.findByRole("option", { name: endpoint, exact: true }));
};

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
    const editor = await openJsonEditor();
    const example = (editor as HTMLTextAreaElement).value;
    const resetButton = screen.getByRole("button", { name: "Reset example" });
    expect(resetButton).toBeDisabled();

    fireEvent.change(editor, { target: { value: "{}" } });
    expect(resetButton).toBeEnabled();
    await user.click(resetButton);

    expect(editor).toHaveValue(example);
    expect(resetButton).toBeDisabled();
  });

  it("offers both literal proxy endpoints without TypeSafe or product prefixes", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    expect(screen.getByRole("combobox", { name: "Decision endpoint" })).toHaveTextContent("/v1/systemone");
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await userEvent.keyboard("{ArrowDown}");
    expect(await screen.findByRole("option", { name: "/v1/decisions", exact: true })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "/v1/systemone", exact: true })).toBeInTheDocument();
    expect(screen.getAllByRole("option")).toHaveLength(2);
    expect(screen.queryByText(/TypeSafe/)).not.toBeInTheDocument();
    await userEvent.keyboard("{Escape}");

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

  it("opens on /v1/systemone with the first decision model the key can call", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);

    await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));
    expect(lookupAuthHeaders(mockModelLookup.mock.calls)).toEqual([expect.arrayContaining(["Bearer session-key"])]);
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(mockFetch.mock.calls[0]?.[0]).not.toMatch(/typesafe/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toMatchObject({
      model: "jev-latest",
      questions: { has_repro_steps: { type: "noul" } },
    });
  });

  it("sends the selected endpoint's real shape and displays the native Decisions response JSON", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));
    const systemOneEditor = await openJsonEditor();
    const systemOneDraft = JSON.stringify({
      model: "jev-latest",
      state: "An outage",
      questions: { urgent: { type: "noul", instructions: "Is it urgent?" } },
    });
    fireEvent.change(systemOneEditor, { target: { value: systemOneDraft } });

    await selectEndpoint("/v1/decisions");
    const editor = screen.getByRole("textbox", { name: "Decisions JSON payload" });
    const nativeExample = (editor as HTMLTextAreaElement).value;
    expect(JSON.parse(nativeExample)).toMatchObject({
      input: expect.any(String),
      questions: expect.arrayContaining([expect.objectContaining({ type: "predicate" })]),
    });
    expect(screen.queryByRole("tab", { name: "Form" })).not.toBeInTheDocument();
    const payload = {
      model: "jev-latest",
      input: "An outage",
      questions: [{ name: "urgent", type: "predicate", instructions: "Is it urgent?" }],
      safety_identifier: "qa-session",
    };
    fireEvent.change(editor, { target: { value: JSON.stringify(payload) } });
    const response: OpenAIDecisionsResponse = {
      model: "jev-latest",
      answers: [{ name: "urgent", type: "predicate", probability: 0.95 }],
      usage: { input_tokens: 10, output_tokens: 1, total_tokens: 11 },
    };
    mockFetch.mockResolvedValueOnce(createResponse(response));
    await user.click(screen.getByRole("button", { name: "Send" }));

    const result = await screen.findByRole("region", { name: "Decisions response JSON" });
    expect(result).toHaveTextContent('"probability": 0.95');
    expect(result).toHaveTextContent('"answers": [');
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/decisions$/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toEqual(payload);

    await selectEndpoint("/v1/systemone");
    expect(screen.getByRole("textbox", { name: "System One JSON payload" })).toHaveValue(systemOneDraft);
    expect(screen.queryByRole("region", { name: "Decisions response JSON" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[1]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(JSON.parse(mockFetch.mock.calls[1]?.[1]?.body as string)).toEqual(JSON.parse(systemOneDraft));

    await selectEndpoint("/v1/decisions");
    expect(screen.getByRole("textbox", { name: "Decisions JSON payload" })).toHaveValue(JSON.stringify(payload));
    await user.click(screen.getByRole("button", { name: "Reset example" }));
    expect(screen.getByRole("textbox", { name: "Decisions JSON payload" })).toHaveValue(nativeExample);
    await selectEndpoint("/v1/systemone");
    expect(screen.getByRole("textbox", { name: "System One JSON payload" })).toHaveValue(systemOneDraft);
  });

  it("validates /v1/decisions independently and preserves an invalid draft across switches", async () => {
    render(<SystemOneUI accessToken="session-key" />);
    await selectEndpoint("/v1/decisions");
    const editor = screen.getByRole("textbox", { name: "Decisions JSON payload" });
    fireEvent.change(editor, { target: { value: "{" } });
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await selectEndpoint("/v1/systemone");
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
    await selectEndpoint("/v1/decisions");
    expect(screen.getByRole("textbox", { name: "Decisions JSON payload" })).toHaveValue("{");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it.each(["success", "error"])("cancels the previous endpoint request and ignores a late %s", async (outcome) => {
    const pending = Promise.withResolvers<Response>();
    mockFetch.mockReturnValueOnce(pending.promise);
    const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    rtlRender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="session-key" />
      </QueryClientProvider>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByRole("button", { name: "Cancel request" });
    await selectEndpoint("/v1/decisions");
    expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    await act(async () => {
      if (outcome === "success") {
        pending.resolve(createResponse({ model: "stale-model", answers: {} }));
      } else {
        pending.reject(new Error("Stale request failed"));
      }
      await pending.promise.catch(() => undefined);
    });
    await waitFor(() => expect(queryClient.isMutating()).toBe(0));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText("stale-model")).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Decisions JSON payload" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
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

    const editor = await openJsonEditor();
    expect(JSON.parse((editor as HTMLTextAreaElement).value)).toMatchObject({
      model: "pplx-decider",
      questions: { has_repro_steps: { type: "noul" } },
    });
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string).model).toBe("pplx-decider");
  });

  it("routes a configured TypeSafe model through the same proxy endpoint", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(
      modelGroupInfoResponse([{ model_group: "typesafe/jev-latest", mode: "evaluation" }]),
    );
    render(<SystemOneUI accessToken="session-key" />);
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("typesafe/jev-latest"),
    );

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(mockFetch.mock.calls[0]?.[0]).not.toMatch(/\/typesafe\//);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string).model).toBe("typesafe/jev-latest");
  });

  it("links to Add Model when the key cannot call any decision model", async () => {
    render(<SystemOneUI accessToken="session-key" />);

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

    await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));
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
    expect(screen.getByRole("combobox", { name: "Decision endpoint" })).toHaveTextContent("/v1/systemone");
  });

  it("keeps an edited request in place when decision models load afterwards", async () => {
    const models = Promise.withResolvers<Response>();
    mockModelLookup.mockReturnValue(models.promise);
    render(<SystemOneUI accessToken="session-key" />);
    const editor = await openJsonEditor();
    const draft = JSON.stringify({ state: "edited", questions: { q: { type: "noul", instructions: "Yes?" } } });
    fireEvent.change(editor, { target: { value: draft } });

    await act(async () => {
      models.resolve(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(editor).toHaveValue(draft);
    expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("");
  });

  it("shows invalid JSON and disables Send", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    fireEvent.change(await openJsonEditor(), {
      target: { value: "{" },
    });

    expect(await screen.findByText(/Invalid JSON syntax/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("sends to /v1/systemone even without discovered models and renders calibrated answers", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem("customProxyBaseUrl", "https://stale.example.com/");
    render(<SystemOneUI accessToken="session-key" />);
    sessionStorage.setItem("customProxyBaseUrl", "https://tenant.example.com/");

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findAllByText("technical")).toHaveLength(2);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, options] = mockFetch.mock.calls[0] ?? [];
    expect(url).toBe("https://tenant.example.com/v1/systemone");
    expect(options?.method).toBe("POST");
    expect(options?.headers).toMatchObject({ "Content-Type": "application/json" });
    expect(Object.values(options?.headers as Record<string, string>)).toContain("Bearer session-key");
    expect(JSON.parse(options?.body as string)).toMatchObject({
      model: "your-decision-model",
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

    expect(await screen.findByText("your-decision-model")).toBeInTheDocument();
    expect(screen.getByText("Selected choice")).toBeInTheDocument();
  });

  it("validates native question values before sending", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    fireEvent.change(await openJsonEditor(), {
      target: {
        value: JSON.stringify({
          state: "A new support request",
          questions: {
            category: {
              type: "choice",
              instructions: 1,
              criteria: { support: true },
            },
          },
        }),
      },
    });

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(screen.getByText("Enter a valid request to preview its state and questions.")).toBeInTheDocument();
  });

  it("sends structured native questions without changing their payload", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    const editor = await openJsonEditor();
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

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toEqual(JSON.parse(draft));
  });

  it("sends a native request without model so the proxy can select its default", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    const payload = { state: "An outage", questions: { urgent: { type: "noul", instructions: "Is this urgent?" } } };
    fireEvent.change(await openJsonEditor(), {
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

  it.each(["success", "error"])("ignores a late %s after editing the request", async (outcome) => {
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
    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), { target: { value: "A new request" } });
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

    expect(await screen.findByRole("alert")).toHaveTextContent("/v1/systemone response has an invalid shape.");
    expect(screen.queryByText("jev-1.13.0")).not.toBeInTheDocument();
  });

  it("clears a previous answer when the request editor changes", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();

    fireEvent.change(await openJsonEditor(), {
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

  describe("request form", () => {
    const question = (name: string) => within(screen.getByRole("group", { name: `Question ${name}` }));
    const sentBody = () => JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string);

    async function pickAnswerType(user: ReturnType<typeof userEvent.setup>, name: string, type: string) {
      question(name).getByRole("combobox", { name: "Answer type" }).focus();
      await user.keyboard("{ArrowDown}");
      await user.click(await screen.findByRole("option", { name: type }));
    }

    const fill = (field: HTMLElement, value: string) => fireEvent.change(field, { target: { value } });

    function rename(field: HTMLElement, name: string) {
      fill(field, name);
      fireEvent.blur(field);
    }

    it("sends a choice and a yes/no question built in the form without opening JSON", async () => {
      const user = userEvent.setup();
      render(<SystemOneUI accessToken="session-key" />);
      const input = screen.getByRole("textbox", { name: "Input" });
      fill(input, "The login page is blank after the upgrade");
      for (const name of ["area", "has_repro_steps", "severity"]) {
        await user.click(screen.getByRole("button", { name: `Remove question ${name}` }));
      }

      await user.click(screen.getByRole("button", { name: "Add question" }));
      rename(question("question_1").getByRole("textbox", { name: "Question name" }), "team");
      fill(question("team").getByRole("textbox", { name: "Instructions" }), "Which team owns this?");
      const [first, second] = question("team").getAllByRole("textbox", { name: "Option label" });
      rename(first, "frontend");
      rename(second, "backend");
      fill(question("team").getByRole("textbox", { name: "Description of frontend" }), "Dashboard and UI");
      await user.click(question("team").getByRole("button", { name: "Add option" }));
      rename(question("team").getAllByRole("textbox", { name: "Option label" })[2], "infra");
      fill(question("team").getByRole("textbox", { name: "Description of infra" }), "Deploys");

      await user.click(screen.getByRole("button", { name: "Add question" }));
      await pickAnswerType(user, "question_1", "Yes / no");
      rename(question("question_1").getByRole("textbox", { name: "Question name" }), "urgent");
      fill(question("urgent").getByRole("textbox", { name: "Instructions" }), "Is this blocking users?");
      fill(question("urgent").getByRole("textbox", { name: "Yes means" }), "Nobody can log in");

      await user.click(screen.getByRole("button", { name: "Send" }));

      expect(await screen.findByText("Selected choice")).toBeInTheDocument();
      expect(screen.queryByRole("textbox", { name: "System One JSON payload" })).not.toBeInTheDocument();
      expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
      expect(sentBody()).toEqual({
        model: "your-decision-model",
        state: "The login page is blank after the upgrade",
        questions: {
          team: {
            type: "choice",
            instructions: "Which team owns this?",
            criteria: { frontend: "Dashboard and UI", backend: "", infra: "Deploys" },
          },
          urgent: { type: "noul", instructions: "Is this blocking users?", criteria: { true: "Nobody can log in" } },
        },
      });
      expect(Object.keys(sentBody().questions)).toEqual(["team", "urgent"]);
      expect(Object.keys(sentBody().questions.team.criteria)).toEqual(["frontend", "backend", "infra"]);
    });

    it("edits score levels and keeps instructions when a question changes type", async () => {
      const user = userEvent.setup();
      mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
      render(<SystemOneUI accessToken="session-key" />);
      await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));

      const level = question("severity").getByRole("textbox", { name: "Level 0" });
      fill(level, "Typo");
      await user.click(question("severity").getByRole("button", { name: "Remove level 4" }));
      await user.click(question("severity").getByRole("button", { name: "Add level" }));
      fill(question("severity").getByRole("textbox", { name: "Level 4" }), "Data loss");
      await pickAnswerType(user, "has_repro_steps", "Choice");
      await user.click(question("has_repro_steps").getByRole("button", { name: "Add option" }));
      await user.click(question("area").getByRole("button", { name: "Remove option docs" }));

      await user.click(screen.getByRole("button", { name: "Send" }));

      expect(await screen.findByText("Selected choice")).toBeInTheDocument();
      expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
      expect(sentBody().model).toBe("jev-latest");
      expect(sentBody().questions.severity.criteria).toEqual([
        "Typo",
        "Minor bug with an easy workaround",
        "Broken feature with a workaround",
        "Broken feature with no workaround",
        "Data loss",
      ]);
      expect(sentBody().questions.has_repro_steps).toEqual({
        type: "choice",
        instructions: "Does the issue include steps someone could follow to reproduce it?",
        criteria: { option_1: "", option_2: "", option_3: "" },
      });
      expect(Object.keys(sentBody().questions.area.criteria)).toEqual(["backend", "sdk", "ui"]);
    });

    it("refuses an empty or taken question name and trims a new one", async () => {
      const user = userEvent.setup();
      render(<SystemOneUI accessToken="session-key" />);
      const name = question("severity").getByRole("textbox", { name: "Question name" });
      expect(name).toHaveAttribute("aria-invalid", "false");

      fill(name, "area");
      expect(name).toHaveAttribute("aria-invalid", "true");
      expect(question("severity").getByText('"area" is already used')).toBeInTheDocument();
      fireEvent.blur(name);
      expect(name).toHaveValue("severity");

      fill(name, "");
      expect(question("severity").getByText("Enter a name")).toBeInTheDocument();
      fireEvent.blur(name);
      expect(name).toHaveValue("severity");

      rename(name, "  impact  ");
      expect(question("impact").getByRole("textbox", { name: "Question name" })).toHaveValue("impact");
      await user.click(screen.getByRole("button", { name: "Add question" }));
      await user.click(screen.getByRole("button", { name: "Add question" }));
      const json = JSON.parse(((await openJsonEditor()) as HTMLTextAreaElement).value);
      expect(Object.keys(json.questions)).toEqual(["area", "has_repro_steps", "impact", "question_1", "question_2"]);
    });

    it("round-trips unknown fields between JSON and the form and points to JSON when it cannot show one", async () => {
      const user = userEvent.setup();
      render(<SystemOneUI accessToken="session-key" />);
      expect(screen.queryByRole("button", { name: "Format JSON" })).not.toBeInTheDocument();

      fireEvent.change(await openJsonEditor(), {
        target: { value: JSON.stringify({ state: { ticket: 1 }, questions: {} }) },
      });
      expect(screen.getByRole("button", { name: "Format JSON" })).toBeInTheDocument();
      await user.click(screen.getByRole("tab", { name: "Form" }));
      expect(screen.getByText(/This request has JSON the form can.t show/)).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Edit in JSON" }));

      const payload = {
        model: "m",
        state: "s",
        metadata: { k: 1 },
        questions: { q: { type: "noul", instructions: "i", note: "x" } },
      };
      fireEvent.change(screen.getByRole("textbox", { name: "System One JSON payload" }), {
        target: { value: JSON.stringify(payload) },
      });
      await user.click(screen.getByRole("tab", { name: "Form" }));
      expect(screen.getByRole("textbox", { name: "Input" })).toHaveValue("s");
      fill(question("q").getByRole("textbox", { name: "Yes means" }), "y");

      const json = JSON.parse(((await openJsonEditor()) as HTMLTextAreaElement).value);
      expect(json).toEqual({ ...payload, questions: { q: { ...payload.questions.q, criteria: { true: "y" } } } });
      expect(Object.keys(json)).toEqual(["model", "state", "metadata", "questions"]);
    });
  });
});
