import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SystemOneUI from "./SystemOneUI";
import type { SystemOneResponse } from "./system_one_types";

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

  it("updates the request editor when a preset is selected", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("combobox", { name: "System One preset" }));
    await user.click(await screen.findByRole("option", { name: "Choice: Support routing" }));

    expect((screen.getByRole("textbox", { name: "System One JSON payload" }) as HTMLTextAreaElement).value).toContain(
      '"department"',
    );
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
    sessionStorage.setItem("customProxyBaseUrl", "https://tenant.example.com/");
    render(<SystemOneUI accessToken="session-key" />);

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
      questions: { needs_human: { type: "noul" } },
    });
  });

  it("renders upstream errors inline", async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce(createResponse(responseBody, 401, "Virtual key rejected"));
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Virtual key rejected");
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
