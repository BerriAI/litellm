import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { normalizeBackendPageUrlToUiOrigin } from "./returnUrlUtils";
import { setServerRootPath } from "@/lib/serverRootPath";
import { navigateToUiPage } from "./uiNavigation";

vi.mock("@/components/networking", () => ({
  getUiPageUrl: (url: string) =>
    normalizeBackendPageUrlToUiOrigin(url, "https://backend.example", "https://ui.example"),
}));

const assign = vi.fn();
const replace = vi.fn();

describe("navigateToUiPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("window", { location: { origin: "https://ui.example", assign, replace } });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setServerRootPath("/");
  });

  it.each([false, true])("uses document navigation with replace=%s", (replaceHistory) => {
    navigateToUiPage("/ui/teams?tab=all#members", replaceHistory);

    expect(replaceHistory ? replace : assign).toHaveBeenCalledWith("https://ui.example/ui/teams?tab=all#members");
    expect(replaceHistory ? assign : replace).not.toHaveBeenCalled();
  });

  it.each([
    ["https://backend.example/ui/teams?tab=all#members", "https://ui.example/ui/teams?tab=all#members"],
    ["https://backend.example//attacker.example/login", "https://ui.example//attacker.example/login"],
    ["https://backend.example/\\attacker.example/login", "https://ui.example//attacker.example/login"],
    ["https://ui.example/ui/teams", "https://ui.example/ui/teams"],
    ["https://attacker.example/ui", "/ui/"],
    ["//attacker.example/ui", "/ui/"],
    ["/\\attacker.example/ui", "/ui/"],
    ["/\t/attacker.example/ui", "/ui/"],
    ["javascript://ui.example/%0Aalert(1)", "/ui/"],
    ["ftp://ui.example/ui", "/ui/"],
    ["https://ui.example:444/ui", "/ui/"],
    ["http://[invalid", "/ui/"],
  ])("navigates safely for %j", (url, expected) => {
    navigateToUiPage(url);

    expect(assign).toHaveBeenCalledWith(expected);
  });

  it("falls back to the dashboard home under the server root path for an unsafe target", () => {
    setServerRootPath("/9000");

    navigateToUiPage("https://attacker.example/ui");

    expect(assign).toHaveBeenCalledWith("/9000/ui/");
  });
});
