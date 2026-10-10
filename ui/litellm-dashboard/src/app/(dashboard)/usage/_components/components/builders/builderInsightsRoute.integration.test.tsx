import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { renderWithProviders, screen, waitFor } from "../../../../../../../tests/test-utils";
import { useBuilderInsightsRoute } from "./builderInsightsRoute";

function RouteControls() {
  const { builderId, selectBuilder, closeBuilder } = useBuilderInsightsRoute();
  return (
    <div>
      <output>{builderId ?? "index"}</output>
      <button type="button" onClick={() => selectBuilder("ava")}>
        Select Ava
      </button>
      <button type="button" onClick={closeBuilder}>
        Close detail
      </button>
    </div>
  );
}

describe("Builder Insights URL selection", () => {
  it("selects and closes a builder query while preserving unrelated URL state", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const lastSearchParams = () => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;
    renderWithProviders(<RouteControls />, { searchParams: "?team=core", onUrlUpdate });

    await user.click(screen.getByRole("button", { name: "Select Ava" }));
    await waitFor(() => expect(lastSearchParams()?.get("builder")).toBe("ava"));
    expect(lastSearchParams()?.get("team")).toBe("core");

    await user.click(screen.getByRole("button", { name: "Close detail" }));
    await waitFor(() => expect(lastSearchParams()?.has("builder")).toBe(false));
    expect(lastSearchParams()?.get("team")).toBe("core");
  });
});
