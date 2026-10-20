import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-ad-hoc-z-index.mjs";

const ruleTester = new RuleTester({
  languageOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    parserOptions: { ecmaFeatures: { jsx: true } },
  },
});

ruleTester.run("no-ad-hoc-z-index", rule as never, {
  valid: [
    'const c = "sticky top-0 z-chrome border-b";',
    'const c = "absolute top-full z-floating mt-1";',
    'const c = "fixed inset-0 z-overlay flex";',
    'const c = "relative z-raised";',
    'const c = "sticky top-0 z-sticky";',
    'const c = "z-sticky-pinned";',
    'const c = "z-auto";',
    'const c = "team-xyz-789";',
    'const c = "bg-gray-50 text-gray-900";',
    "const c = `flex ${open ? 'z-overlay' : ''}`;",
    'const el = <div className="absolute z-floating" />;',
    "const style = { position: 'fixed', top: 0 };",
    { code: 'const c = "isolate z-popup";', options: [{ allowPopupLayer: true }] },
    { code: 'const c = "data-[side=top]:z-popup";', options: [{ allowPopupLayer: true }] },
    { code: 'const c = "[&:hover]:z-popup";', options: [{ allowPopupLayer: true }] },
    'const c = "data-[side=top]:z-floating";',
    'const c = "[&>[data-z-50]]:z-overlay";',
    'const c = "z-overlay!";',
  ],
  invalid: [
    { code: 'const c = "fixed inset-0 z-50";', errors: [{ messageId: "adHoc", data: { token: "z-50" } }] },
    { code: 'const c = "z-10";', errors: [{ messageId: "adHoc", data: { token: "z-10" } }] },
    { code: 'const c = "z-0";', errors: [{ messageId: "adHoc", data: { token: "z-0" } }] },
    { code: 'const c = "-z-10";', errors: [{ messageId: "adHoc", data: { token: "-z-10" } }] },
    { code: 'const c = "z-[1100]";', errors: [{ messageId: "adHoc", data: { token: "z-[1100]" } }] },
    { code: 'const c = "z-9999";', errors: [{ messageId: "adHoc", data: { token: "z-9999" } }] },
    { code: 'const c = "z-(--my-z)";', errors: [{ messageId: "adHoc", data: { token: "z-(--my-z)" } }] },
    { code: 'const c = "md:z-50";', errors: [{ messageId: "adHoc", data: { token: "md:z-50" } }] },
    { code: 'const c = "hover:md:z-[2]";', errors: [{ messageId: "adHoc", data: { token: "hover:md:z-[2]" } }] },
    {
      code: 'const c = "data-[side=top]:z-50";',
      errors: [{ messageId: "adHoc", data: { token: "data-[side=top]:z-50" } }],
    },
    { code: 'const c = "[&>*]:z-[5]";', errors: [{ messageId: "adHoc", data: { token: "[&>*]:z-[5]" } }] },
    { code: 'const c = "[&:hover]:z-10";', errors: [{ messageId: "adHoc", data: { token: "[&:hover]:z-10" } }] },
    {
      code: 'const c = "group-data-[state=open]:z-(--x)";',
      errors: [{ messageId: "adHoc", data: { token: "group-data-[state=open]:z-(--x)" } }],
    },
    { code: 'const c = "!z-50";', errors: [{ messageId: "adHoc", data: { token: "!z-50" } }] },
    { code: 'const c = "md:z-50!";', errors: [{ messageId: "adHoc", data: { token: "md:z-50!" } }] },
    {
      code: 'const c = "data-[side=top]:z-popup";',
      errors: [{ messageId: "popupReserved", data: { token: "data-[side=top]:z-popup" } }],
    },
    {
      code: "const c = `max-h-full ${extra} z-[1100]`;",
      errors: [{ messageId: "adHoc", data: { token: "z-[1100]" } }],
    },
    {
      code: 'const el = <div className="fixed inset-0 z-50" />;',
      errors: [{ messageId: "adHoc", data: { token: "z-50" } }],
    },
    {
      code: 'const c = "z-10 md:z-20";',
      errors: [{ messageId: "adHoc" }, { messageId: "adHoc" }],
    },
    { code: 'const c = "isolate z-popup";', errors: [{ messageId: "popupReserved", data: { token: "z-popup" } }] },
    {
      code: 'const c = "isolate z-popup";',
      options: [{ allowPopupLayer: false }],
      errors: [{ messageId: "popupReserved", data: { token: "z-popup" } }],
    },
    { code: "const style = { position: 'fixed', zIndex: 1000 };", errors: [{ messageId: "inlineZIndex" }] },
    { code: "const style = { 'z-index': 1000 };", errors: [{ messageId: "inlineZIndex" }] },
    {
      code: "const el = <div style={{ zIndex: 40 }} />;",
      errors: [{ messageId: "inlineZIndex" }],
    },
    {
      code: 'const c = "z-50";',
      options: [{ allowPopupLayer: true }],
      errors: [{ messageId: "adHoc", data: { token: "z-50" } }],
    },
  ],
});
