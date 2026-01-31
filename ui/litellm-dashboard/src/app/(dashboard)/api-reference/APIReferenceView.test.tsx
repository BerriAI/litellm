import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import APIReferenceView from "./APIReferenceView";

vi.mock("./components/CodeBlock", () => ({
  __esModule: true,
  default: ({ code }: { code: string }) => <pre data-testid="api-reference-code-block">{code}</pre>,
}));

describe("APIReferenceView", () => {
  const codeBlockTestId = "api-reference-code-block";

  it("uses the API doc base url when provided", () => {
    const apiDocUrl = "https://docs.litellm.test";
    const { getAllByTestId } = render(<APIReferenceView proxySettings={{ LITELLM_UI_API_DOC_BASE_URL: apiDocUrl }} />);

    const codeBlocks = getAllByTestId(codeBlockTestId);
    expect(codeBlocks[0].textContent).toContain(apiDocUrl);
  });

  it("falls back to the proxy base url when the docs url is missing", () => {
    const proxyUrl = "https://proxy.litellm.test";
    const { getAllByTestId } = render(<APIReferenceView proxySettings={{ PROXY_BASE_URL: proxyUrl }} />);

    const codeBlocks = getAllByTestId(codeBlockTestId);
    expect(codeBlocks[0].textContent).toContain(proxyUrl);
  });

  it("prefers the docs url when both urls are provided", () => {
    const apiDocUrl = "https://docs-preferred.litellm.test";
    const proxyUrl = "https://proxy-backup.litellm.test";

    const { getAllByTestId } = render(
      <APIReferenceView
        proxySettings={{
          LITELLM_UI_API_DOC_BASE_URL: apiDocUrl,
          PROXY_BASE_URL: proxyUrl,
        }}
      />,
    );

    const codeBlocks = getAllByTestId(codeBlockTestId);
    const renderedCode = codeBlocks[0].textContent ?? "";
    expect(renderedCode).toContain(apiDocUrl);
    expect(renderedCode).not.toContain(proxyUrl);
  });
});
