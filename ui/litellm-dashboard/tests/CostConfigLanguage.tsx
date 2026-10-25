import { fireEvent, renderHook, screen } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { LanguageProvider } from "@/contexts/LanguageContext";

// 费用 hook 使用真实语言与通知树，测试只替换网络边界。
function CostConfigLanguage({ children }: PropsWithChildren) {
  return (
    <LanguageProvider defaultLanguage="en" userId="cost-config" storageScope="test">
      <LanguageSelector />
      {children}
      <Toaster expand />
    </LanguageProvider>
  );
}

export const renderCostConfigHook = <T,>(hook: () => T) => renderHook(hook, { wrapper: CostConfigLanguage });

export const changeCostConfigLanguage = (value: "en" | "zh-CN") =>
  fireEvent.change(screen.getByRole("combobox", { name: /Interface language|界面语言/ }), { target: { value } });
