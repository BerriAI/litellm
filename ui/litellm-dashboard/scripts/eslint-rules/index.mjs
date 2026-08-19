import noLargeInlineObjectArg from "./no-large-inline-object-arg.mjs";
import noLongConditionChain from "./no-long-condition-chain.mjs";
import noComplexJsxArrow from "./no-complex-jsx-arrow.mjs";
import filenamePascalCase from "./filename-pascal-case.mjs";
import noAntdClassSelectors from "./no-antd-class-selectors.mjs";
import noClickByText from "./no-click-by-text.mjs";

const plugin = {
  rules: {
    "no-large-inline-object-arg": noLargeInlineObjectArg,
    "no-long-condition-chain": noLongConditionChain,
    "no-complex-jsx-arrow": noComplexJsxArrow,
    "filename-pascal-case": filenamePascalCase,
    "no-antd-class-selectors": noAntdClassSelectors,
    "no-click-by-text": noClickByText,
  },
};

export default plugin;
