import { test, expect, Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

/**
 * ChatUI E2E Tests
 *
 * These tests verify the ChatUI playground component renders and behaves correctly.
 * They are designed to run before AND after a refactor to ensure no regressions.
 *
 * The tests cover:
 * 1. Page layout & structure (sidebar, chat area, controls)
 * 2. Endpoint switching and conditional UI
 * 3. Model selector behavior
 * 4. Chat input and suggested prompts
 * 5. Clear chat functionality
 * 6. Get Code modal
 * 7. API key source switching
 * 8. Custom Proxy Base URL
 * 9. Conditional sections (voice, agent, MCP, image upload, audio upload)
 */

test.describe("ChatUI Playground", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  /** Navigate to the playground and wait for it to load */
  async function goToPlayground(page: PlaywrightPage) {
    await navigateToPage(page, Page.LlmPlayground);
    // Suppress popups/banners
    await page.evaluate(() => {
      window.localStorage.setItem("disableUsageIndicator", "true");
      window.localStorage.setItem("disableShowPrompts", "true");
      window.localStorage.setItem("disableShowNewBadge", "true");
    });
    // Wait for the sidebar to appear
    await expect(page.getByText("Configurations", { exact: true })).toBeVisible({ timeout: 15000 });
    // Dismiss any overlays
    for (const label of ["Dismiss", "Don't ask me again"]) {
      const btn = page.getByRole("button", { name: label });
      if (await btn.isVisible({ timeout: 500 }).catch(() => false)) {
        await btn.click();
      }
    }
  }

  /**
   * Open the endpoint dropdown and select an option.
   * Uses the Ant Design Select's wrapper div to click (avoids combobox interception issues).
   */
  async function switchEndpoint(page: PlaywrightPage, endpointTitle: string) {
    // Find the select that currently shows a /v1/ or /mcp- endpoint value
    const endpointSelectItem = page.locator(".ant-select-selection-item").filter({
      hasText: /^\//,
    }).first();
    await endpointSelectItem.click();
    await page.getByTitle(endpointTitle, { exact: true }).click();
    // Wait for the new endpoint to be reflected
    await expect(
      page.locator(".ant-select-selection-item").filter({ hasText: endpointTitle }).first(),
    ).toBeVisible();
  }

  // ──────────────────────────────────────────────
  // 1. Page Layout & Structure
  // ──────────────────────────────────────────────

  test.describe("Layout & Structure", () => {
    test("renders the playground page with sidebar and chat area", async ({ page }) => {
      await goToPlayground(page);

      // Left sidebar elements
      await expect(page.getByText("Configurations", { exact: true })).toBeVisible();
      await expect(page.getByText("Virtual Key Source").first()).toBeVisible();
      await expect(page.getByText("Custom Proxy Base URL").first()).toBeVisible();
      await expect(page.getByText("Endpoint Type").first()).toBeVisible();
      await expect(page.getByText("Select Model").first()).toBeVisible();

      // Main chat area
      await expect(page.getByText("Test Key").first()).toBeVisible();
      await expect(page.getByRole("button", { name: /Clear Chat/ })).toBeVisible();
      await expect(page.getByRole("button", { name: /Get Code/ })).toBeVisible();

      // Empty state message
      await expect(
        page.getByText("Start a conversation, generate an image, or handle audio"),
      ).toBeVisible();
    });

    test("shows suggested prompts when chat is empty", async ({ page }) => {
      await goToPlayground(page);

      await expect(page.getByRole("button", { name: "Write me a poem" })).toBeVisible();
      await expect(page.getByRole("button", { name: "Explain quantum computing" })).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Draft a polite email requesting a meeting" }),
      ).toBeVisible();
    });

    test("has a text input area with correct placeholder", async ({ page }) => {
      await goToPlayground(page);

      await expect(
        page.getByRole("textbox", { name: /Type your message/ }),
      ).toBeVisible();
    });

    test("sidebar shows Tags section", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("Tags", { exact: true }).first()).toBeVisible();
    });

    test("sidebar shows MCP Servers section", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("MCP Servers").first()).toBeVisible();
    });

    test("sidebar shows Vector Store section", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("Vector Store").first()).toBeVisible();
    });

    test("sidebar shows Guardrails section", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("Guardrails", { exact: true }).first()).toBeVisible();
    });

    test("sidebar shows Policies section", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("Policies", { exact: true }).first()).toBeVisible();
    });

    test("shows Chat tab as selected by default", async ({ page }) => {
      await goToPlayground(page);
      const chatTab = page.getByRole("tab", { name: "Chat" });
      await expect(chatTab).toBeVisible();
      await expect(chatTab).toHaveAttribute("aria-selected", "true");
    });

    test("shows Compare and Compliance tabs", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByRole("tab", { name: "Compare" })).toBeVisible();
      await expect(page.getByRole("tab", { name: "Compliance" })).toBeVisible();
    });
  });

  // ──────────────────────────────────────────────
  // 2. Endpoint Switching
  // ──────────────────────────────────────────────

  test.describe("Endpoint Switching", () => {
    test("defaults to /v1/chat/completions endpoint", async ({ page }) => {
      await goToPlayground(page);

      await expect(
        page.locator(".ant-select-selection-item").filter({ hasText: "/v1/chat/completions" }).first(),
      ).toBeVisible();
    });

    test("can switch to /v1/responses endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/responses");

      await expect(
        page.locator(".ant-select-selection-item").filter({ hasText: "/v1/responses" }).first(),
      ).toBeVisible();
    });

    test("shows voice selector only for audio/speech endpoint", async ({ page }) => {
      await goToPlayground(page);

      // Voice should NOT be visible on default chat endpoint
      await expect(page.getByText("Voice", { exact: true })).not.toBeVisible();

      await switchEndpoint(page, "/v1/audio/speech");

      // Now Voice selector should be visible
      await expect(page.getByText("Voice", { exact: true })).toBeVisible();
    });

    test("shows agent selector only for A2A endpoint", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("Select Agent")).not.toBeVisible();

      await switchEndpoint(page, "/v1/a2a/message/send");
      await expect(page.getByText("Select Agent")).toBeVisible();
    });

    test("hides model selector for A2A endpoint", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("Select Model").first()).toBeVisible();

      await switchEndpoint(page, "/v1/a2a/message/send");
      await expect(page.getByText("Select Model")).not.toBeVisible();
    });

    test("hides model selector for MCP direct endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/mcp-rest/tools/call");
      await expect(page.getByText("Select Model")).not.toBeVisible();
    });

    test("shows image upload area for image edits endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/images/edits");
      await expect(page.getByText("Click or drag images to upload")).toBeVisible();
    });

    test("shows audio upload area for transcription endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/audio/transcriptions");
      await expect(page.getByText("Click or drag audio file to upload")).toBeVisible();
    });

    test("changes placeholder text for image generation endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/images/generations");
      await expect(
        page.getByRole("textbox", { name: /Describe the image you want to generate/ }),
      ).toBeVisible();
    });

    test("changes placeholder text for speech endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/audio/speech");
      await expect(
        page.getByRole("textbox", { name: /Enter text to convert to speech/ }),
      ).toBeVisible();
    });

    test("shows A2A suggested prompts for A2A endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/a2a/message/send");

      await expect(page.getByRole("button", { name: "What can you help me with?" })).toBeVisible();
      await expect(page.getByRole("button", { name: "Tell me about yourself" })).toBeVisible();
      await expect(page.getByRole("button", { name: "What tasks can you perform?" })).toBeVisible();
    });

    test("MCP label changes from 'MCP Servers' to 'MCP Server' for MCP endpoint", async ({ page }) => {
      await goToPlayground(page);
      await expect(page.getByText("MCP Servers").first()).toBeVisible();

      await switchEndpoint(page, "/mcp-rest/tools/call");
      // The text should now be singular (with trailing space or exact match)
      await expect(page.locator("text=MCP Server").first()).toBeVisible();
    });

    test("can switch to each major endpoint type", async ({ page }) => {
      await goToPlayground(page);

      // Verify we can successfully switch to several different endpoint types
      // This implicitly validates they exist in the dropdown
      const endpointsToTest = [
        "/v1/responses",
        "/v1/images/generations",
        "/v1/audio/speech",
        "/v1/embeddings",
      ];

      for (const endpoint of endpointsToTest) {
        await switchEndpoint(page, endpoint);
        await expect(
          page.locator(".ant-select-selection-item").filter({ hasText: endpoint }).first(),
        ).toBeVisible();
      }
    });
  });

  // ──────────────────────────────────────────────
  // 3. Model Selector
  // ──────────────────────────────────────────────

  test.describe("Model Selector", () => {
    test("shows 'Enter custom model' option in model dropdown", async ({ page }) => {
      await goToPlayground(page);

      // Click the model selector's wrapper div (avoids combobox input intercepting)
      const modelSelect = page.locator(".ant-select-selector").filter({
        has: page.locator(".ant-select-selection-placeholder", { hasText: "Select a Model" }),
      }).first();
      await modelSelect.click({ force: true });

      await expect(page.getByText("Enter custom model")).toBeVisible();
    });

    test("shows custom model input when 'Enter custom model' is selected", async ({ page }) => {
      await goToPlayground(page);

      const modelSelect = page.locator(".ant-select-selector").filter({
        has: page.locator(".ant-select-selection-placeholder", { hasText: "Select a Model" }),
      }).first();
      await modelSelect.click({ force: true });
      await page.getByText("Enter custom model").click();

      await expect(page.getByPlaceholder("Enter custom model name")).toBeVisible();
    });
  });

  // ──────────────────────────────────────────────
  // 4. Chat Input & Suggested Prompts
  // ──────────────────────────────────────────────

  test.describe("Chat Input", () => {
    test("clicking a suggested prompt fills the input", async ({ page }) => {
      await goToPlayground(page);

      await page.getByRole("button", { name: "Write me a poem" }).click();

      const textarea = page.getByRole("textbox", { name: /Type your message/ });
      await expect(textarea).toHaveValue("Write me a poem");
    });

    test("send button is disabled when input is empty", async ({ page }) => {
      await goToPlayground(page);

      const sendButton = page.getByRole("button", { name: "arrow-up" });
      await expect(sendButton).toBeDisabled();
    });

    test("send button becomes enabled when input has text", async ({ page }) => {
      await goToPlayground(page);

      const textarea = page.getByRole("textbox", { name: /Type your message/ });
      await textarea.fill("Hello world");

      const sendButton = page.getByRole("button", { name: "arrow-up" });
      await expect(sendButton).toBeEnabled();
    });
  });

  // ──────────────────────────────────────────────
  // 5. Clear Chat
  // ──────────────────────────────────────────────

  test.describe("Clear Chat", () => {
    test("clear chat button is visible and clickable", async ({ page }) => {
      await goToPlayground(page);

      const clearBtn = page.getByRole("button", { name: /Clear Chat/ });
      await expect(clearBtn).toBeVisible();
      await clearBtn.click();

      // Should show success notification
      await expect(page.getByText("Chat history cleared.")).toBeVisible({ timeout: 5000 });
    });
  });

  // ──────────────────────────────────────────────
  // 6. Get Code Modal
  // ──────────────────────────────────────────────

  test.describe("Get Code Modal", () => {
    test("opens code modal when Get Code is clicked", async ({ page }) => {
      await goToPlayground(page);
      await page.getByRole("button", { name: /Get Code/ }).click();

      await expect(page.getByText("Generated Code")).toBeVisible();
      await expect(page.getByText("SDK Type")).toBeVisible();
      await expect(page.getByRole("button", { name: /Copy to Clipboard/ })).toBeVisible();
    });

    test("code modal defaults to OpenAI SDK", async ({ page }) => {
      await goToPlayground(page);
      await page.getByRole("button", { name: /Get Code/ }).click();

      await expect(
        page.locator(".ant-select-selection-item").filter({ hasText: "OpenAI SDK" }),
      ).toBeVisible();
    });

    test("code modal can switch to Azure SDK", async ({ page }) => {
      await goToPlayground(page);
      await page.getByRole("button", { name: /Get Code/ }).click();
      await expect(page.getByText("Generated Code")).toBeVisible();

      // Click the SDK type selector (shows "OpenAI SDK")
      const sdkSelectItem = page.locator(".ant-select-selection-item").filter({
        hasText: "OpenAI SDK",
      });
      await sdkSelectItem.click();

      // Select Azure SDK from the dropdown
      await page.getByTitle("Azure SDK", { exact: true }).click();

      await expect(
        page.locator(".ant-select-selection-item").filter({ hasText: "Azure SDK" }),
      ).toBeVisible();
    });

    test("code modal can be closed", async ({ page }) => {
      await goToPlayground(page);
      await page.getByRole("button", { name: /Get Code/ }).click();
      await expect(page.getByText("Generated Code")).toBeVisible();

      // Close via the X button on the modal
      const closeBtn = page.locator(".ant-modal-close").first();
      await closeBtn.click();
      await expect(page.getByText("Generated Code")).not.toBeVisible();
    });
  });

  // ──────────────────────────────────────────────
  // 7. API Key Source
  // ──────────────────────────────────────────────

  test.describe("API Key Source", () => {
    test("defaults to Current UI Session", async ({ page }) => {
      await goToPlayground(page);

      await expect(
        page.locator(".ant-select-selection-item").filter({ hasText: "Current UI Session" }).first(),
      ).toBeVisible();
    });

    test("switching to Virtual Key shows custom key input", async ({ page }) => {
      await goToPlayground(page);

      // Click the "Current UI Session" text in the selector
      const currentSessionItem = page.locator(".ant-select-selection-item").filter({
        hasText: "Current UI Session",
      }).first();
      await currentSessionItem.click();
      await page.getByTitle("Virtual Key", { exact: true }).click();

      await expect(page.getByPlaceholder("Enter custom Virtual Key")).toBeVisible();
    });
  });

  // ──────────────────────────────────────────────
  // 8. Custom Proxy Base URL
  // ──────────────────────────────────────────────

  test.describe("Custom Proxy Base URL", () => {
    test("shows custom proxy URL input", async ({ page }) => {
      await goToPlayground(page);
      await expect(
        page.getByRole("textbox", { name: /Enter custom proxy URL/ }),
      ).toBeVisible();
    });

    test("entering custom proxy URL shows confirmation text", async ({ page }) => {
      await goToPlayground(page);

      const proxyInput = page.getByRole("textbox", { name: /Enter custom proxy URL/ });
      await proxyInput.fill("http://my-proxy:8080");

      await expect(page.getByText("API calls will be sent to: http://my-proxy:8080")).toBeVisible();
    });
  });

  // ──────────────────────────────────────────────
  // 9. Code Interpreter (Responses endpoint)
  // ──────────────────────────────────────────────

  test.describe("Code Interpreter", () => {
    test("code interpreter section is visible for responses endpoint", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/responses");

      await expect(page.getByText("Code Interpreter").first()).toBeVisible();
    });
  });

  // ──────────────────────────────────────────────
  // 10. Endpoint-specific Feature Combinations
  // ──────────────────────────────────────────────

  test.describe("Endpoint Feature Combinations", () => {
    test("switching endpoints resets model selection", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/responses");

      // The model select should show "Select a Model" placeholder (not a selected value)
      const modelPlaceholder = page.locator(".ant-select-selection-placeholder").filter({
        hasText: "Select a Model",
      });
      await expect(modelPlaceholder.first()).toBeVisible();
    });

    test("embeddings endpoint shows correct placeholder", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/embeddings");

      await expect(
        page.getByRole("textbox", { name: /Type your message/ }),
      ).toBeVisible();
    });

    test("image edits endpoint shows edit description placeholder", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/images/edits");

      await expect(
        page.getByRole("textbox", { name: /Describe how you want to edit the image/ }),
      ).toBeVisible();
    });

    test("transcription endpoint has disabled send when no audio uploaded", async ({ page }) => {
      await goToPlayground(page);
      await switchEndpoint(page, "/v1/audio/transcriptions");

      const sendButton = page.getByRole("button", { name: "arrow-up" });
      await expect(sendButton).toBeDisabled();
    });
  });
});
