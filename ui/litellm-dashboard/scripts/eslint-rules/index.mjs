import noLargeInlineObjectArg from "./no-large-inline-object-arg.mjs";
import noLongConditionChain from "./no-long-condition-chain.mjs";
import noComplexJsxArrow from "./no-complex-jsx-arrow.mjs";
import filenamePascalCase from "./filename-pascal-case.mjs";
import noAntdImport from "./no-antd-import.mjs";

const plugin = {
  rules: {
    "no-large-inline-object-arg": noLargeInlineObjectArg,
    "no-long-condition-chain": noLongConditionChain,
    "no-complex-jsx-arrow": noComplexJsxArrow,
    "filename-pascal-case": filenamePascalCase,
    "no-antd-import": noAntdImport,
  },
};

export default plugin;
