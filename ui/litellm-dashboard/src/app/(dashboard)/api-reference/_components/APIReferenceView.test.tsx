import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import APIReferenceView from "./APIReferenceView";

vi.mock("@/components/CodeBlock", () => ({
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

  it("renders the page title, blurb and docs link", () => {
    render(<APIReferenceView proxySettings={{ PROXY_BASE_URL: "https://proxy.litellm.test" }} />);

    expect(screen.getByText("OpenAI Compatible Proxy: API Reference")).toBeTruthy();
    expect(screen.getByText(/LiteLLM is OpenAI Compatible/)).toBeTruthy();

    const docsLink = screen.getByRole("link", { name: /API Reference Docs/ });
    expect(docsLink.getAttribute("href")).toBe("https://docs.litellm.ai/docs/proxy/user_keys");
    expect(docsLink.getAttribute("target")).toBe("_blank");
  });

  it("exposes the three SDK tabs with the first selected by default", () => {
    render(<APIReferenceView proxySettings={{ PROXY_BASE_URL: "https://proxy.litellm.test" }} />);

    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "OpenAI Python SDK",
      "LlamaIndex",
      "Langchain Py",
    ]);
    expect(screen.getAllByRole("tab").map((tab) => tab.getAttribute("aria-selected"))).toEqual([
      "true",
      "false",
      "false",
    ]);
  });

  it.each([
    ["OpenAI Python SDK", "import openai"],
    ["LlamaIndex", "from llama_index.llms import AzureOpenAI"],
    ["Langchain Py", "from langchain.chat_models import ChatOpenAI"],
  ])("selecting %s shows its snippet wired to the base url", async (tabName, marker) => {
    const proxyUrl = "https://proxy.litellm.test";
    const user = userEvent.setup();
    render(<APIReferenceView proxySettings={{ PROXY_BASE_URL: proxyUrl }} />);

    await user.click(screen.getByRole("tab", { name: tabName }));

    expect(screen.getByRole("tab", { name: tabName }).getAttribute("aria-selected")).toBe("true");

    const selectedPanel = screen.getByRole("tabpanel");
    expect(selectedPanel.textContent).toContain(marker);
    expect(selectedPanel.textContent).toContain(proxyUrl);
  });
});
