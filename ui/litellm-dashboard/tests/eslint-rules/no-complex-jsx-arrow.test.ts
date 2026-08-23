import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-complex-jsx-arrow.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module", parserOptions: { ecmaFeatures: { jsx: true } } },
});

ruleTester.run("no-complex-jsx-arrow", rule as never, {
  valid: [
    "const x = <button onClick={() => doThing()} />;",
    "const x = <button onClick={() => { a(); }} />;",
    "const x = <button onClick={() => { a(); b(); }} />;",
    "const handler = () => { a(); b(); c(); }; const x = <button onClick={handler} />;",
    "const run = () => { a(); b(); c(); };",
    "foo(() => { a(); b(); c(); });",
    "const x = <List renderItem={(i) => i.name} />;",
    { code: "const x = <button onClick={() => { a(); b(); c(); }} />;", options: [{ maxStatements: 3 }] },
  ],
  invalid: [
    {
      code: "const x = <button onClick={() => { a(); b(); c(); }} />;",
      errors: [{ messageId: "tooComplex", data: { count: 3, max: 2 } }],
    },
    {
      code: "const x = <form onSubmit={() => { a(); b(); c(); d(); }} />;",
      errors: [{ messageId: "tooComplex", data: { count: 4, max: 2 } }],
    },
    {
      code: "const x = <button onClick={() => { a(); b(); }} />;",
      options: [{ maxStatements: 1 }],
      errors: [{ messageId: "tooComplex", data: { count: 2, max: 1 } }],
    },
  ],
});
