import { act, fireEvent, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { SemanticCacheJsonControls } from "./SemanticCacheControls";

afterEach(() => {
  testQueryClient.clear();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("clears the override without removing other metadata", () => {
  const onChange = vi.fn();
  renderWithProviders(
    <SemanticCacheJsonControls value='{"cache_params":{"enabled":false},"region":"us"}' onChange={onChange} />,
  );
  const mode = screen.getByLabelText("Cache mode");
  expect(mode).toHaveValue("false");
  fireEvent.change(mode, { target: { value: "" } });
  expect(JSON.parse(onChange.mock.calls[0][0])).toEqual({ region: "us" });
});

it("preserves disabled mode and a saved threshold while editing the TTL for an exact cache", async () => {
  vi.useFakeTimers();
  const fetchSettings = vi.fn().mockResolvedValue(jsonResponse({ supports_similarity_threshold: false }));
  vi.stubGlobal("fetch", fetchSettings);
  const onChange = vi.fn();
  renderWithProviders(
    <SemanticCacheJsonControls
      value='{"cache_params":{"enabled":false,"similarity_threshold":0,"ttl":null},"region":"us"}'
      onChange={onChange}
    />,
  );
  expect(screen.getByLabelText("Cache mode")).toHaveValue("false");
  await act(async () => {
    await vi.runAllTimersAsync();
  });
  expect(fetchSettings).toHaveBeenCalledWith(expect.stringMatching(/\/get\/ui_settings$/), { method: "GET" });
  expect(screen.queryByLabelText("Similarity threshold")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("TTL in seconds"), { target: { value: "60" } });
  expect(JSON.parse(onChange.mock.calls[0][0])).toEqual({
    cache_params: { enabled: false, similarity_threshold: 0, ttl: 60 },
    region: "us",
  });
});

it("shows the saved similarity threshold when the active cache supports it", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ supports_similarity_threshold: true })));
  renderWithProviders(
    <SemanticCacheJsonControls value='{"cache_params":{"similarity_threshold":0.8}}' onChange={vi.fn()} />,
  );
  expect(await screen.findByLabelText("Similarity threshold")).toHaveValue(0.8);
});
