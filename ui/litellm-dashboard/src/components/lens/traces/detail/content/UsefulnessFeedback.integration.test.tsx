import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import { readRequest } from "@/../tests/lens-test-utils";
import type { components } from "@/lib/http/schema";

import { UsefulnessFeedback } from "./UsefulnessFeedback";

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  proxyBaseUrl: "https://gateway.example",
}));

afterEach(() => vi.unstubAllGlobals());

function renderFeedback() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <UsefulnessFeedback accessToken="test" traceId="trace" traceRef="ref" spanId="span" />
    </QueryClientProvider>,
  );
}

it("submits zero, reloads its saved value, updates without increasing the count, and removes it", async () => {
  const user = userEvent.setup();
  let mine: components["schemas"]["Feedback"] | null = null;
  const bodies: unknown[] = [];
  let revision = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = await readRequest(input, init);
      if (request.method === "POST") {
        bodies.push(request.body);
        revision += 1;
        mine = {
          ...(request.body as components["schemas"]["FeedbackCreate"]),
          id: "feedback",
          created_at: "2026-01-01",
          updated_at: String(revision),
        };
        return Response.json(mine);
      }
      if (request.method === "DELETE") {
        mine = null;
        return new Response(null, { status: 204 });
      }
      const summary = {
        key: "usefulness",
        count: mine ? 1 : 0,
        average: mine?.value ?? null,
        mine,
        can_rate: true,
        distribution: { "0": mine?.value === 0 ? 1 : 0, "10": mine?.value === 10 ? 1 : 0 },
      };
      return Response.json(summary);
    }),
  );
  renderFeedback();
  expect(await screen.findByText("Unrated")).toBeVisible();
  expect(screen.getByRole("button", { name: "Save rating" })).toBeDisabled();
  await user.click(screen.getByRole("radio", { name: "0 out of 10" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Comment (optional)" }), { target: { value: "Not useful" } });
  await user.click(screen.getByRole("button", { name: "Save rating" }));
  expect(await screen.findByText("0.0 / 10 · 1 rating")).toBeVisible();
  expect(bodies).toEqual([
    { trace_id: "trace", trace_ref: "ref", span_id: "span", key: "usefulness", value: 0, comment: "Not useful" },
  ]);
  expect(screen.getByRole("radio", { name: "0 out of 10" })).toBeChecked();
  await user.click(screen.getByRole("radio", { name: "10 out of 10" }));
  await user.click(screen.getByRole("button", { name: "Update rating" }));
  expect(await screen.findByText("10.0 / 10 · 1 rating")).toBeVisible();
  await user.click(screen.getByText("Score distribution"));
  expect(screen.getByText("10: 1")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Remove rating" }));
  expect(await screen.findByText("Unrated")).toBeVisible();
  expect(screen.getByRole("button", { name: "Save rating" })).toBeDisabled();
});

it("shows a submission error without pretending the score was saved", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = await readRequest(input, init);
      if (request.method === "POST")
        return Response.json({ detail: "Read-only users cannot submit feedback" }, { status: 403 });
      const summary = { count: 0, average: null, mine: null, distribution: {}, can_rate: true };
      return Response.json(summary);
    }),
  );
  renderFeedback();
  await user.click(await screen.findByRole("radio", { name: "8 out of 10" }));
  await user.click(screen.getByRole("button", { name: "Save rating" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Read-only users cannot submit feedback");
  expect(screen.getByText("Unrated")).toBeVisible();
});

it("shows aggregates but no write controls for a read-only reviewer", async () => {
  const summary = { count: 0, average: null, mine: null, distribution: {}, can_rate: false };
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => Response.json(summary)),
  );
  renderFeedback();
  expect(await screen.findByText("Unrated")).toBeVisible();
  expect(screen.queryByRole("button", { name: "Save rating" })).not.toBeInTheDocument();
  expect(screen.getByText(/Read-only access/)).toBeVisible();
});
