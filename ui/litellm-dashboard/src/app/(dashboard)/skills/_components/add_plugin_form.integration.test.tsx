import React from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import AddPluginForm from "./add_plugin_form";
import { registerClaudeCodePlugin } from "@/components/networking";
import { toast } from "@/lib/toast";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { translate } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";

vi.unmock("@/lib/toast");

vi.mock("@/components/networking", () => ({
  registerClaudeCodePlugin: vi.fn().mockResolvedValue({ status: "success" }),
}));

const mockRegister = vi.mocked(registerClaudeCodePlugin);

const DEFAULT_PROPS = {
  visible: true,
  onClose: vi.fn(),
  accessToken: "sk-test",
  onSuccess: vi.fn(),
};

const URL_PLACEHOLDER = "https://github.com/org/repo or https://bucket.s3.amazonaws.com/my-skill.zip";
const SUBPATH_PLACEHOLDER = "plugins/my-skill";
const SHA256_PLACEHOLDER = "64 hex characters";
const S3_ZIP_URL = "https://skills-bucket.s3.us-east-1.amazonaws.com/plugins/s3-skill-1.0.0.zip";
const DIGEST = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

describe("AddPluginForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    toast.dismiss();
  });

  afterEach(() => act(() => toast.dismiss()));

  it("已显示的技能名称校验切换语言后立即更新", async () => {
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="skill-errors" storageScope="test">
        <LanguageSelector />
        <AddPluginForm {...DEFAULT_PROPS} />
      </LanguageProvider>,
    );
    await typeUrl("https://github.com/org/home");
    fireEvent.change(screen.getByPlaceholderText("my-skill"), { target: { value: "Home invalid" } });
    await submit();
    expect(await screen.findByText("Name must be kebab-case (lowercase, numbers, hyphens only)")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByText("名称必须使用 kebab-case 格式，只能包含小写字母、数字和连字符")).toBeVisible();
    expect(screen.getByDisplayValue("Home invalid")).toBeVisible();
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it.each([
    ["https://github.com/org/home/tree/main/plugins/home", "该 URL 已指向子目录，因此此字段已禁用"],
    ["https://example.com/home.zip", "ZIP 压缩包将整体安装，因此此字段已禁用"],
  ])("来源 %s 的禁用说明跟随语言，不修改来源", async (url, description) => {
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="skill-lock" storageScope="test">
        <LanguageSelector />
        <AddPluginForm {...DEFAULT_PROPS} />
      </LanguageProvider>,
    );
    await typeUrl(url);
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByText(description)).toBeVisible();
    expect(screen.getByDisplayValue(url)).toBeVisible();
    expect(screen.getByPlaceholderText(SUBPATH_PLACEHOLDER)).toBeDisabled();
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it.each(["success", "fallback", "backend"] as const)(
    "异步注册 %s 通知使用当前语言，服务端文本保持原文",
    async (result) => {
      const pending = Promise.withResolvers<Awaited<ReturnType<typeof registerClaudeCodePlugin>>>();
      mockRegister.mockReturnValueOnce(pending.promise);
      renderWithProviders(
        <LanguageProvider defaultLanguage="en" userId="skill-notices" storageScope="test">
          <LanguageSelector />
          <AddPluginForm {...DEFAULT_PROPS} />
          <Toaster expand />
        </LanguageProvider>,
      );
      await typeUrl("https://github.com/org/home");
      await submit();
      expect(mockRegister).toHaveBeenCalledOnce();
      fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
      await act(async () => {
        if (result === "success") pending.resolve({ status: "success" });
        else pending.reject(new Error(result === "backend" ? "Home" : ""));
      });
      const messages = {
        success: { chinese: "技能已注册", english: "Skill registered successfully" },
        fallback: { chinese: "注册技能失败", english: "Failed to register skill" },
        backend: { chinese: "Home", english: "Home" },
      };
      expect(await screen.findByText(messages[result].chinese, { exact: true })).toBeVisible();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      expect(screen.getByText(messages[result].english, { exact: true })).toBeVisible();
      expect(mockRegister).toHaveBeenCalledOnce();
    },
  );

  it("切换语言保留技能来源与名称草稿，并显示中文校验", async () => {
    localStorage.clear();
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="skill-validation" storageScope="test">
        <LanguageSelector />
        <AddPluginForm {...DEFAULT_PROPS} />
      </LanguageProvider>,
    );
    const sourceUrl = "https://github.com/anthropics/my-skill";
    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(URL_PLACEHOLDER), { target: { value: sourceUrl } });
    });
    fireEvent.change(screen.getByPlaceholderText("my-skill"), { target: { value: "Home invalid" } });

    // 改变语言后应保留来源信息，且不发起任何技能注册请求。
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByDisplayValue(sourceUrl)).toBeInTheDocument();
    expect(screen.getByDisplayValue("Home invalid")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: translate("zh-CN", "Add Skill") }));
    expect(await screen.findByText("名称必须使用 kebab-case 格式，只能包含小写字母、数字和连字符")).toBeInTheDocument();
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it("renders the host-agnostic source URL input and subfolder field, hiding the digest until a zip is entered", () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    expect(screen.getByText("Source URL")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(URL_PLACEHOLDER)).toBeInTheDocument();
    expect(screen.getByText("Subfolder path (Optional)")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(SUBPATH_PLACEHOLDER)).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(SHA256_PLACEHOLDER)).not.toBeInTheDocument();
  });

  it("shows GitHub repo preview for a plain repo URL", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(URL_PLACEHOLDER);

    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://github.com/anthropics/claude-code" },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/GitHub repo/)).toBeInTheDocument();
    });
  });

  it("shows git-subdir preview for a tree URL and disables the subfolder field", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(URL_PLACEHOLDER);

    await act(async () => {
      fireEvent.change(urlInput, {
        target: {
          value: "https://github.com/anthropics/claude-code/tree/main/plugins/my-skill",
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/GitHub subdir/)).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText(SUBPATH_PLACEHOLDER)).toBeDisabled();
  });

  it("shows a raw url preview for a non-github host", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(URL_PLACEHOLDER);

    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://gitlab.com/group/repo" },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/Git repo/)).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText(SUBPATH_PLACEHOLDER)).toBeEnabled();
  });

  it("combines a repo URL with a subfolder into a git-subdir preview", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(URL_PLACEHOLDER);
    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://gitlab.com/group/repo" },
      });
    });

    const subPathInput = screen.getByPlaceholderText(SUBPATH_PLACEHOLDER);
    await act(async () => {
      fireEvent.change(subPathInput, { target: { value: "plugins/x" } });
    });

    await waitFor(() => {
      expect(screen.getByText(/Git subdir/)).toBeInTheDocument();
      expect(screen.getByText(/plugins\/x/)).toBeInTheDocument();
    });
  });

  it("auto-fills skill name from repo URL", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(URL_PLACEHOLDER);

    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://github.com/anthropics/my-awesome-skill" },
      });
    });

    await waitFor(() => {
      const nameInput = screen.getByPlaceholderText("my-skill") as HTMLInputElement;
      expect(nameInput.value).toBe("my-awesome-skill");
    });
  });

  it("does not auto-fill name when name is already set", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const nameInput = screen.getByPlaceholderText("my-skill") as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: "existing-name" } });

    const urlInput = screen.getByPlaceholderText(URL_PLACEHOLDER);

    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://github.com/anthropics/other-skill" },
      });
    });

    await waitFor(() => {
      expect(nameInput.value).toBe("existing-name");
    });
  });

  const typeUrl = async (value: string) => {
    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(URL_PLACEHOLDER), { target: { value } });
    });
  };

  const typeSubPath = async (value: string) => {
    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(SUBPATH_PLACEHOLDER), { target: { value } });
    });
  };

  const submit = async () => {
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Add Skill" }));
    });
  };

  it("submits a github repo source", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl("https://github.com/anthropics/claude-code");
    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({ source: { source: "github", repo: "anthropics/claude-code" } }),
      );
    });
  });

  it("submits a github subdir source from a tree URL", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl("https://github.com/anthropics/claude-code/tree/main/plugins/my-skill");
    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({
          source: { source: "git-subdir", url: "https://github.com/anthropics/claude-code", path: "plugins/my-skill" },
        }),
      );
    });
  });

  it("submits a raw url source for a gitlab repo", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl("https://gitlab.com/group/repo");
    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({ source: { source: "url", url: "https://gitlab.com/group/repo" } }),
      );
    });
  });

  it("submits a git-subdir source from a gitlab repo plus subfolder field", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl("https://gitlab.com/group/repo");
    await typeSubPath("plugins/x");
    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({
          source: { source: "git-subdir", url: "https://gitlab.com/group/repo", path: "plugins/x" },
        }),
      );
    });
  });

  it("clears the subfolder field and uses the URL path once a tree URL is entered", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeSubPath("plugins/x");
    const subPathInput = screen.getByPlaceholderText(SUBPATH_PLACEHOLDER) as HTMLInputElement;
    expect(subPathInput.value).toBe("plugins/x");

    await typeUrl("https://github.com/anthropics/claude-code/tree/main/plugins/from-url");

    await waitFor(() => {
      expect(subPathInput.value).toBe("");
      expect(subPathInput).toBeDisabled();
    });

    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({
          source: {
            source: "git-subdir",
            url: "https://github.com/anthropics/claude-code",
            path: "plugins/from-url",
          },
        }),
      );
    });
  });

  it("shows a zip archive preview, disables the subfolder field, and reveals the digest field", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl(S3_ZIP_URL);

    expect(await screen.findByText(/Zip archive/)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(SUBPATH_PLACEHOLDER)).toBeDisabled();
    expect(screen.getByText("A zip archive is installed as a whole, so this field is disabled")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(SHA256_PLACEHOLDER)).toBeInTheDocument();
    expect((screen.getByPlaceholderText("my-skill") as HTMLInputElement).value).toBe("s3-skill-1-0-0");
  });

  it("submits an archive source without a digest when the field is left empty", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl(S3_ZIP_URL);
    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({ source: { source: "archive", url: S3_ZIP_URL } }),
      );
    });
  });

  it("submits an archive source pinned to the lowercased digest", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl(S3_ZIP_URL);
    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(SHA256_PLACEHOLDER), {
        target: { value: ` ${DIGEST.toUpperCase()} ` },
      });
    });
    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({ source: { source: "archive", url: S3_ZIP_URL, sha256: DIGEST } }),
      );
    });
  });

  it("drops the digest once the archive URL changes so a stale checksum is never sent for a new file", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl(S3_ZIP_URL);
    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(SHA256_PLACEHOLDER), { target: { value: DIGEST } });
    });
    const otherZipUrl = S3_ZIP_URL.replace("1.0.0", "1.1.0");
    await typeUrl(otherZipUrl);

    expect(screen.getByPlaceholderText(SHA256_PLACEHOLDER)).toHaveValue("");
    await submit();

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "sk-test",
        expect.objectContaining({ source: { source: "archive", url: otherZipUrl } }),
      );
    });
  });

  it("blocks submission and shows the digest error for a malformed sha256", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    await typeUrl(S3_ZIP_URL);
    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(SHA256_PLACEHOLDER), { target: { value: "not-a-digest" } });
    });
    await submit();

    expect(await screen.findByText("SHA-256 must be a 64-character hex digest")).toBeInTheDocument();
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it("surfaces the backend error message when registration fails", async () => {
    mockRegister.mockRejectedValueOnce(new Error("Plugin 'claude-code' already exists"));
    renderWithProviders(
      <>
        <AddPluginForm {...DEFAULT_PROPS} />
        <Toaster expand />
      </>,
    );

    await typeUrl("https://github.com/anthropics/claude-code");
    await submit();

    expect(await screen.findByText("Plugin 'claude-code' already exists", { exact: true })).toBeVisible();
  });

  it("surfaces the 409 name-conflict reason verbatim without burying it under a generic failure prefix", async () => {
    const conflictMessage =
      "A skill named 'gitlab' already exists. Update the existing skill instead of adding it again.";
    mockRegister.mockRejectedValueOnce(new Error(conflictMessage));
    renderWithProviders(
      <>
        <AddPluginForm {...DEFAULT_PROPS} />
        <Toaster expand />
      </>,
    );

    await typeUrl("https://github.com/anthropics/claude-code");
    await submit();

    expect(await screen.findByText(conflictMessage, { exact: true })).toBeVisible();
    expect(mockRegister).toHaveBeenCalledOnce();
  });
});
