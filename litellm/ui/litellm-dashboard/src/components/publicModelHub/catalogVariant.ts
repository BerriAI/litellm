import plugin from "tailwindcss/plugin";

import { narrow } from "./catalogBreakpoints.json";

export default plugin(({ addVariant }) => {
  addVariant("catalog-narrow", `@media ${narrow}`);
});
