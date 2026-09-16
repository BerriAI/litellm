import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../../../../tests/test-utils";
import APIReferenceView from "./APIReferenceView";

vi.mock("@/components/CodeBlock", () => ({
  __esModule: true,
  default: ({ code }: { code: string }) => <pre data-testid="api-reference-code-block">{code}</pre>,
}));

describe("APIReferenceView", () => {
  const codeBlockTestId = "api-reference-code-block";

  it("uses the API doc base url when provided", () => {
    const apiDocUrl = "https://docs.litellm.test";
    renderWithProviders(<APIReferenceView proxySettings={{ LITELLM_UI_API_DOC_BASE_URL: apiDocUrl }} />);

    const codeBlocks = screen.getAllByTestId(codeBlockTestId);
    expect(codeBlocks[0]).toHaveTextContent(new RegExp(apiDocUrl));
  });

  it("falls back to the proxy base url when the docs url is missing", () => {
    const proxyUrl = "https://proxy.litellm.test";
    renderWithProviders(<APIReferenceView proxySettings={{ PROXY_BASE_URL: proxyUrl }} />);

    const codeBlocks = screen.getAllByTestId(codeBlockTestId);
    expect(codeBlocks[0]).toHaveTextContent(new RegExp(proxyUrl));
  });

  it("prefers the docs url when both urls are provided", () => {
    const apiDocUrl = "https://docs-preferred.litellm.test";
    const proxyUrl = "https://proxy-backup.litellm.test";

    renderWithProviders(
      <APIReferenceView
        proxySettings={{
          LITELLM_UI_API_DOC_BASE_URL: apiDocUrl,
          PROXY_BASE_URL: proxyUrl,
        }}
      />,
    );

    const codeBlocks = screen.getAllByTestId(codeBlockTestId);
    const renderedCode = codeBlocks[0].textContent ?? "";
    expect(renderedCode).toContain(apiDocUrl);
    expect(renderedCode).not.toContain(proxyUrl);
  });

  it("renders the page title, blurb and docs link", () => {
    renderWithProviders(<APIReferenceView proxySettings={{ PROXY_BASE_URL: "https://proxy.litellm.test" }} />);

    expect(screen.getByText("OpenAI Compatible Proxy: API Reference")).toBeInTheDocument();
    expect(screen.getByText(/LiteLLM is OpenAI Compatible/)).toBeInTheDocument();

    const docsLink = screen.getByRole("link", { name: /API Reference Docs/ });
    expect(docsLink).toHaveAttribute("href", "https://docs.litellm.ai/docs/proxy/user_keys");
    expect(docsLink).toHaveAttribute("target", "_blank");
  });

  it("exposes the three SDK tabs with the first selected by default", () => {
    renderWithProviders(<APIReferenceView proxySettings={{ PROXY_BASE_URL: "https://proxy.litellm.test" }} />);

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
    renderWithProviders(<APIReferenceView proxySettings={{ PROXY_BASE_URL: proxyUrl }} />);

    await user.click(screen.getByRole("tab", { name: tabName }));

    expect(screen.getByRole("tab", { name: tabName })).toHaveAttribute("aria-selected", "true");

    const selectedPanel = screen.getByRole("tabpanel");
    expect(selectedPanel).toHaveTextContent(new RegExp(marker));
    expect(selectedPanel).toHaveTextContent(new RegExp(proxyUrl));
  });

  it("opens the SDK named in the URL", () => {
    renderWithProviders(<APIReferenceView proxySettings={{ PROXY_BASE_URL: "https://proxy.litellm.test" }} />, {
      searchParams: "?sdk=langchain",
    });

    expect(screen.getByRole("tab", { name: "Langchain Py" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toHaveTextContent(/from langchain.chat_models import ChatOpenAI/);
  });

  it("writes the selected SDK to the URL and removes it for the default SDK", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<APIReferenceView proxySettings={{ PROXY_BASE_URL: "https://proxy.litellm.test" }} />, {
      onUrlUpdate,
    });

    await user.click(screen.getByRole("tab", { name: "LlamaIndex" }));
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("sdk")).toBe("llamaindex");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("tab")).toBe(false);

    await user.click(screen.getByRole("tab", { name: "OpenAI Python SDK" }));
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("sdk")).toBe(false);
  });
});
