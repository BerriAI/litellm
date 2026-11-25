import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-arbitrary-design-value.mjs";

const ruleTester = new RuleTester({
  languageOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    parserOptions: { ecmaFeatures: { jsx: true } },
  },
});

const arbitrary = (code: string, token: string) => ({ code, errors: [{ messageId: "arbitrary", data: { token } }] });

ruleTester.run("no-arbitrary-design-value", rule as never, {
  valid: [
    'const c = "text-xs leading-tight tracking-wider rounded-sm border";',
    'const c = "grid-cols-[minmax(0,1fr)_auto] w-[180px] left-[calc(50%-0.5px)]";',
    'const c = "data-[side=right]:sm:max-w-[90vw] [&>p]:mb-0";',
    'const c = "ease-[cubic-bezier(0.4,0,0.2,1)]";',
    'const el = <div className="text-sm text-muted-foreground" />;',
  ],
  invalid: [
    arbitrary('const c = "flex text-[13px]";', "text-[13px]"),
    arbitrary('const c = "text-[#e5484d]";', "text-[#e5484d]"),
    arbitrary('const c = "tracking-[-0.26px]";', "tracking-[-0.26px]"),
    arbitrary('const c = "leading-[1.2]";', "leading-[1.2]"),
    arbitrary('const c = "rounded-[4px]";', "rounded-[4px]"),
    arbitrary('const c = "rounded-t-[4px]";', "rounded-t-[4px]"),
    arbitrary('const c = "border-[0.67px]";', "border-[0.67px]"),
    arbitrary('const c = "[border-left-width:1px]";', "[border-left-width:1px]"),
    arbitrary('const c = "dark:text-[#8b9bff]";', "dark:text-[#8b9bff]"),
    arbitrary('const c = "data-[side=top]:text-[11px]";', "data-[side=top]:text-[11px]"),
    arbitrary("const c = `px-2 ${x} text-[10px]`;", "text-[10px]"),
    arbitrary('const el = <div className="text-[12px]" />;', "text-[12px]"),
  ],
});
