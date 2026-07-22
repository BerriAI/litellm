import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DrawerHeader } from "./DrawerHeader";
import { LogEntry } from "../columns";

const makeLog = (overrides: Partial<LogEntry> = {}): LogEntry =>
  ({
    request_id: "req-123",
    api_key: "",
    team_id: "",
    model: "gpt-4o",
    model_id: "",
    call_type: "acompletion",
    custom_llm_provider: "",
    spend: 0,
    total_tokens: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    startTime: "2026-07-08T10:00:00.000Z",
    endTime: "2026-07-08T10:00:01.000Z",
    cache_hit: "false",
    messages: [],
    response: {},
    metadata: {},
    ...overrides,
  }) as LogEntry;

const renderHeader = (shareUrl: string) =>
  render(
    <DrawerHeader
      log={makeLog()}
      onClose={() => {}}
      onPrevious={() => {}}
      onNext={() => {}}
      statusLabel="Success"
      statusColor="success"
      environment="default"
      shareUrl={shareUrl}
    />,
  );

describe("DrawerHeader share link", () => {
  const writeText = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    writeText.mockClear();
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
  });

  it("copies the share url to the clipboard when the copy-link button is clicked", async () => {
    const shareUrl = "https://proxy.example.com/ui/?page=logs&request_id=req-123";
    renderHeader(shareUrl);

    fireEvent.click(screen.getByRole("button", { name: "Copy link to this log" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(shareUrl));
  });

  it("shows a 'Copied!' tooltip after copying", async () => {
    renderHeader("https://proxy.example.com/ui/?request_id=req-123");

    fireEvent.click(screen.getByRole("button", { name: "Copy link to this log" }));

    await waitFor(() => expect(screen.getByRole("img", { name: "check" })).toBeInTheDocument());
  });
});
