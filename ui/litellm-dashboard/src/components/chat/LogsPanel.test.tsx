import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LogsPanel from "./LogsPanel";
import { renderWithProviders } from "../../../tests/test-utils";
import { uiSpendLogDetailsCall, uiSpendLogsCall } from "../networking";

const translationState = vi.hoisted(() => ({ language: "en" as "en" | "ru" }));

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  return {
    useTranslation: () => ({
      t: (key: string, values?: Record<string, string | number>) => {
        const translation = key.split(".").reduce<unknown>((copy, segment) => {
          if (typeof copy !== "object" || copy === null) return undefined;
          return (copy as Record<string, unknown>)[segment];
        }, resources[translationState.language].chat);
        if (typeof translation !== "string") return key;
        return Object.entries(values ?? {}).reduce(
          (copy, [name, value]) => copy.replaceAll(`{{${name}}}`, String(value)),
          translation,
        );
      },
      i18n: { language: translationState.language, resolvedLanguage: translationState.language },
    }),
  };
});

vi.mock("../networking", () => ({
  uiSpendLogsCall: vi.fn(),
  uiSpendLogDetailsCall: vi.fn(),
}));

const mockedLogsCall = vi.mocked(uiSpendLogsCall);
const mockedDetailsCall = vi.mocked(uiSpendLogDetailsCall);

const sampleRow = {
  request_id: "req-abc-123",
  model: "gpt-4o",
  status: "success",
  spend: 0.0123,
  total_tokens: 1500,
  prompt_tokens: 1000,
  completion_tokens: 500,
  startTime: "2026-07-18T10:00:00Z",
  endTime: "2026-07-18T10:00:02Z",
  request_duration_ms: 2000,
};

const paginated = (rows: unknown[]) => ({
  data: rows,
  total: rows.length,
  page: 1,
  page_size: 50,
  total_pages: rows.length > 0 ? 1 : 0,
});

describe("LogsPanel", () => {
  beforeEach(() => {
    translationState.language = "en";
    vi.clearAllMocks();
    mockedLogsCall.mockResolvedValue(paginated([sampleRow]));
    mockedDetailsCall.mockResolvedValue({ messages: [{ role: "user", content: "hi" }], response: { ok: true } });
  });

  it("renders the log panel in Russian", async () => {
    translationState.language = "ru";
    renderWithProviders(<LogsPanel accessToken="tok-ru" userId="user-1" />);

    expect(await screen.findByText("Ваши логи")).toBeInTheDocument();
    expect(await screen.findByText("Успешно")).toBeInTheDocument();
    expect(await screen.findByRole("columnheader", { name: "Стоимость" })).toBeInTheDocument();
  });

  it("scopes the query to the current user so it only shows their own logs", async () => {
    renderWithProviders(<LogsPanel accessToken="tok-scope" userId="user-42" />);

    await waitFor(() => expect(mockedLogsCall).toHaveBeenCalled());
    expect(mockedLogsCall).toHaveBeenCalledWith(
      expect.objectContaining({
        accessToken: "tok-scope",
        params: expect.objectContaining({ user_id: "user-42" }),
      }),
    );
  });

  it("renders a row for each returned log", async () => {
    renderWithProviders(<LogsPanel accessToken="tok-rows" userId="user-1" />);

    expect(await screen.findByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("1,500")).toBeInTheDocument();
    expect(screen.getByText("Success")).toBeInTheDocument();
  });

  it("shows an empty state when there are no logs", async () => {
    mockedLogsCall.mockResolvedValue(paginated([]));
    renderWithProviders(<LogsPanel accessToken="tok-empty" userId="user-1" />);

    expect(await screen.findByText("No logs for this period")).toBeInTheDocument();
  });

  it("opens the detail dialog and lazily loads request/response when a row is clicked", async () => {
    renderWithProviders(<LogsPanel accessToken="tok-detail" userId="user-1" />);

    const modelCell = await screen.findByText("gpt-4o");
    expect(mockedDetailsCall).not.toHaveBeenCalled();

    fireEvent.click(modelCell);

    expect(await screen.findByText("Request details")).toBeInTheDocument();
    await waitFor(() =>
      expect(mockedDetailsCall).toHaveBeenCalledWith("tok-detail", "req-abc-123", expect.any(String)),
    );
  });

  it("shows an error state (not the empty state) when the logs query fails", async () => {
    mockedLogsCall.mockRejectedValue(new Error("boom"));
    renderWithProviders(<LogsPanel accessToken="tok-err" userId="user-1" />);

    expect(await screen.findByText("Failed to load your logs")).toBeInTheDocument();
    expect(screen.queryByText("No logs for this period")).not.toBeInTheDocument();
  });

  it("falls back to proxy_server_request when messages is empty for the request payload", async () => {
    mockedDetailsCall.mockResolvedValue({
      messages: {},
      proxy_server_request: { body: { messages: [{ role: "user", content: "hello from proxy" }] } },
      response: { ok: true },
    });
    renderWithProviders(<LogsPanel accessToken="tok-fallback" userId="user-1" />);

    fireEvent.click(await screen.findByText("gpt-4o"));

    expect(await screen.findByText(/hello from proxy/)).toBeInTheDocument();
  });
});
