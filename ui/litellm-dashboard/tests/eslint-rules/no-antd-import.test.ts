import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-antd-import.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module" },
});

ruleTester.run("no-antd-import", rule as never, {
  valid: [
    "import { Button } from '@/components/ui/button';",
    "import x from 'antdesign';",
    "import x from 'not-antd';",
    "const x = require('@/lib/foo');",
    "export { Foo } from './Foo';",
  ],
  invalid: [
    { code: "import { Button } from 'antd';", errors: [{ messageId: "antdImport" }] },
    { code: "import Button from 'antd/es/button';", errors: [{ messageId: "antdImport" }] },
    { code: "import 'antd/dist/reset.css';", errors: [{ messageId: "antdImport" }] },
    { code: "export { Button } from 'antd';", errors: [{ messageId: "antdImport" }] },
    { code: "export * from 'antd/es/button';", errors: [{ messageId: "antdImport" }] },
    { code: "const x = require('antd');", errors: [{ messageId: "antdImport" }] },
    { code: "const p = import('antd');", errors: [{ messageId: "antdImport" }] },
  ],
});
