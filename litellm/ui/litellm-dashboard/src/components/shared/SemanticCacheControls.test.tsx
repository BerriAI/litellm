import { fireEvent, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { SemanticCacheJsonControls } from "./SemanticCacheControls";

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

it("preserves a disabled cache when editing a TTL stored as null", () => {
  const onChange = vi.fn();
  renderWithProviders(
    <SemanticCacheJsonControls
      value='{"cache_params":{"enabled":false,"similarity_threshold":0,"ttl":null},"region":"us"}'
      onChange={onChange}
    />,
  );
  expect(screen.getByLabelText("Cache mode")).toHaveValue("false");
  fireEvent.change(screen.getByLabelText("TTL in seconds"), { target: { value: "60" } });
  expect(JSON.parse(onChange.mock.calls[0][0])).toEqual({
    cache_params: { enabled: false, similarity_threshold: 0, ttl: 60 },
    region: "us",
  });
});
