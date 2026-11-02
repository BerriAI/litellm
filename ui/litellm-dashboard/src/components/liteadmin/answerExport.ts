import { ARIA_AGENT_LABEL } from "./aria-brand";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const HTML_ELEMENTS = new Set([
  "article",
  "section",
  "div",
  "span",
  "p",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "strong",
  "em",
  "b",
  "i",
  "ul",
  "ol",
  "li",
  "blockquote",
  "pre",
  "code",
  "table",
  "thead",
  "tbody",
  "tfoot",
  "tr",
  "th",
  "td",
  "br",
  "hr",
  "figure",
  "figcaption",
  "a",
  "del",
  "s",
  "sup",
  "sub",
]);
const SVG_ELEMENTS = new Set([
  "svg",
  "g",
  "path",
  "rect",
  "circle",
  "ellipse",
  "line",
  "polyline",
  "polygon",
  "text",
  "tspan",
  "defs",
  "clipPath",
  "linearGradient",
  "radialGradient",
  "stop",
  "title",
  "desc",
]);
const SVG_ATTRIBUTES = new Set([
  "id",
  "viewBox",
  "width",
  "height",
  "x",
  "y",
  "x1",
  "x2",
  "y1",
  "y2",
  "cx",
  "cy",
  "r",
  "rx",
  "ry",
  "d",
  "points",
  "transform",
  "dx",
  "dy",
  "textLength",
  "lengthAdjust",
  "offset",
  "gradientUnits",
  "gradientTransform",
  "spreadMethod",
  "fx",
  "fy",
  "fr",
  "clipPathUnits",
  "preserveAspectRatio",
]);
const SVG_STYLES = [
  "fill",
  "fill-opacity",
  "fill-rule",
  "stroke",
  "stroke-width",
  "stroke-opacity",
  "stroke-dasharray",
  "stroke-linecap",
  "stroke-linejoin",
  "opacity",
  "font-size",
  "font-family",
  "font-weight",
  "text-anchor",
  "dominant-baseline",
  "stop-color",
  "stop-opacity",
  "clip-path",
] as const;
const EXCLUDED =
  "[data-export-exclude],button,input,textarea,select,script,style,link,meta,iframe,object,embed,img,video,audio,source,canvas,form,[role='tooltip'],.recharts-tooltip-wrapper,.recharts-tooltip-cursor";

const PRINT_CSS = `
  @page { margin: 18mm; }
  * { box-sizing: border-box; }
  body { margin: 0; color: #111827; background: #fff; font: 11pt/1.55 system-ui, sans-serif; }
  header { margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #d1d5db; }
  h1 { margin: 0 0 12px; font-size: 22pt; }
  h2, h3, h4, h5, h6 { margin: 20px 0 8px; break-after: avoid; line-height: 1.3; }
  h2 { font-size: 17pt; } h3 { font-size: 14pt; }
  p { margin: 8px 0; white-space: pre-wrap; overflow-wrap: anywhere; }
  ul, ol { padding-left: 24px; } li { margin: 5px 0; }
  table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 10pt; }
  thead { display: table-header-group; }
  th, td { padding: 8px; border-bottom: 1px solid #d1d5db; text-align: left; overflow-wrap: anywhere; }
  th { background: #f3f4f6; } tr, svg { break-inside: avoid; }
  figcaption { break-after: avoid; }
  blockquote { margin-left: 0; padding-left: 16px; border-left: 3px solid #d1d5db; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 12px; background: #f3f4f6; }
  code { font-family: ui-monospace, monospace; font-size: 0.9em; }
  svg { display: block; max-width: 100%; height: auto; overflow: visible; }
  svg text { fill: #334155; }
  [data-report-chart] { margin: 18px 0; }
  body { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
`;

function safeSvgStyle(value: string): boolean {
  return !/url\s*\(/i.test(value) || /^url\(["']?#[\w:.-]+["']?\)$/.test(value);
}

function copySvgAttributes(source: Element, copy: Element): void {
  for (const attribute of source.attributes) {
    if (SVG_ATTRIBUTES.has(attribute.name)) copy.setAttribute(attribute.name, attribute.value);
  }
  const computed = source.ownerDocument.defaultView?.getComputedStyle(source);
  for (const property of SVG_STYLES) {
    const value = computed?.getPropertyValue(property) || source.getAttribute(property);
    if (value && safeSvgStyle(value)) copy.setAttribute(property, value);
  }
}

function copyAnswerNode(source: Node, target: Document): Node | null {
  if (source.nodeType === Node.TEXT_NODE) return target.createTextNode(source.textContent ?? "");
  if (source.nodeType !== Node.ELEMENT_NODE) return null;
  const element = source as Element;
  const svg = element.namespaceURI === SVG_NAMESPACE;
  const hidden = element.hasAttribute("hidden") || (!svg && element.getAttribute("aria-hidden") === "true");
  if (element.matches(EXCLUDED) || hidden) return null;
  if (svg && !SVG_ELEMENTS.has(element.localName)) return null;

  const copy = svg
    ? target.createElementNS(SVG_NAMESPACE, element.localName)
    : target.createElement(HTML_ELEMENTS.has(element.localName) ? element.localName : "div");
  if (svg) {
    copySvgAttributes(element, copy);
  } else {
    for (const attribute of ["colspan", "rowspan", "start"]) {
      const value = element.getAttribute(attribute);
      if (value && /^\d+$/.test(value)) copy.setAttribute(attribute, value);
    }
    if (element.getAttribute("data-slot") === "chart") copy.setAttribute("data-report-chart", "");
  }
  for (const child of element.childNodes) {
    const safeChild = copyAnswerNode(child, target);
    if (safeChild) copy.appendChild(safeChild);
  }
  return copy;
}

export function createAriaAnswerDocument(question: string, answerElement: HTMLElement): Document {
  const report = answerElement.ownerDocument.implementation.createHTMLDocument(`${ARIA_AGENT_LABEL} answer`);
  report.documentElement.lang = "en";
  const policy = report.createElement("meta");
  policy.httpEquiv = "Content-Security-Policy";
  policy.content = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'";
  report.head.prepend(policy);
  const style = report.createElement("style");
  style.textContent = PRINT_CSS;
  report.head.appendChild(style);
  const header = report.createElement("header");
  const title = report.createElement("h1");
  title.textContent = ARIA_AGENT_LABEL;
  const prompt = report.createElement("p");
  prompt.textContent = question;
  header.append(title, prompt);
  report.body.appendChild(header);
  const answer = copyAnswerNode(answerElement, report);
  if (answer) report.body.appendChild(answer);
  return report;
}

export function exportAriaAnswer(question: string, answerElement: HTMLElement): boolean {
  const owner = answerElement.ownerDocument;
  const report = createAriaAnswerDocument(question, answerElement);
  const frame = owner.createElement("iframe");
  frame.title = `${ARIA_AGENT_LABEL} print preview`;
  frame.setAttribute("sandbox", "allow-modals allow-same-origin");
  frame.style.cssText = "position:fixed;inset:0;width:0;height:0;border:0;";
  owner.body.appendChild(frame);
  const printWindow = frame.contentWindow;
  const printDocument = frame.contentDocument;
  if (!printWindow || !printDocument || !owner.defaultView) {
    frame.remove();
    return false;
  }
  printDocument.replaceChild(printDocument.importNode(report.documentElement, true), printDocument.documentElement);
  const cleanup = () => {
    owner.defaultView?.clearTimeout(cleanupTimer);
    printWindow.removeEventListener("afterprint", cleanup);
    frame.remove();
  };
  printWindow.addEventListener("afterprint", cleanup, { once: true });
  const cleanupTimer = owner.defaultView.setTimeout(cleanup, 300_000);
  try {
    printWindow.focus();
    printWindow.print();
    return true;
  } catch {
    cleanup();
    return false;
  }
}
