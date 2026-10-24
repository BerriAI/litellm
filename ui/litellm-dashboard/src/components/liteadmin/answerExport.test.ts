// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createAriaAnswerDocument, exportAriaAnswer } from "./answerExport";

function answer(html: string): HTMLElement {
  const element = document.createElement("article");
  element.innerHTML = html;
  document.body.appendChild(element);
  return element;
}

function previewReport(report: Document): Document {
  const frame = document.createElement("iframe");
  document.body.appendChild(frame);
  const preview = frame.contentDocument!;
  preview.replaceChild(preview.importNode(report.documentElement, true), preview.documentElement);
  return preview;
}

afterEach(() => {
  document.body.replaceChildren();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("createAriaAnswerDocument", () => {
  it("should preserve the question, readable answer, table and chart without exporting other messages", () => {
    answer("<p>Another conversation's private receipt</p>");
    const source = answer(`<h2>Usage this week</h2><p>Total spend is <strong>$3.25</strong>.</p>
      <ol><li>Review expensive models</li></ol>
      <table><thead><tr><th>Model</th><th>Spend</th></tr></thead><tbody><tr><td>Primary</td><td>$3.25</td></tr></tbody></table>
      <svg viewBox="0 0 400 200" width="400" height="200"><rect x="20" y="40" width="80" height="120" style="fill: rgb(59, 130, 246);stroke: rgb(10, 20, 30)"/><text x="20" y="185">Primary</text></svg>`);
    const report = previewReport(createAriaAnswerDocument("Which models cost the most?", source));

    expect(report.title).toBe("Aria Agent answer");
    expect(report.body).toHaveTextContent("Which models cost the most?");
    expect(report.getElementsByTagName("h2")[0]).toHaveTextContent("Usage this week");
    expect(report.getElementsByTagName("strong")[0]).toHaveTextContent("$3.25");
    expect(report.getElementsByTagName("li")[0]).toHaveTextContent("Review expensive models");
    expect(report.getElementsByTagName("td")[1]).toHaveTextContent("$3.25");
    expect(report.getElementsByTagName("svg")[0]).toHaveAttribute("viewBox", "0 0 400 200");
    expect(report.getElementsByTagName("rect")[0]).toHaveAttribute("fill", "rgb(59, 130, 246)");
    expect(report.getElementsByTagName("rect")[0]).toHaveAttribute("stroke", "rgb(10, 20, 30)");
    expect(report.body).not.toHaveTextContent("private receipt");
    expect(source.getElementsByTagName("rect")[0]).toHaveStyle({ fill: "rgb(59, 130, 246)" });
  });

  it("should treat question text literally and remove executable markup, resource URLs and export controls", () => {
    const source = answer(`<p onclick="steal()">Safe answer <a href="javascript:steal()">linked text</a></p>
      <button>Copy answer</button><aside data-export-exclude>Secret action receipt</aside>
      <script>steal()</script><img src="https://evil.example/track" onerror="steal()">
      <iframe src="https://evil.example"></iframe><style>@import 'https://evil.example/style';</style>
      <div role="tooltip">Hidden tooltip</div><span aria-hidden="true">Decoration</span>
      <svg viewBox="0 0 100 100" onload="steal()"><foreignObject><div>Injected HTML</div></foreignObject>
      <image href="https://evil.example/track"/><rect width="50" height="50" fill="url(https://evil.example/paint)"/>
      <path d="M0 0 L5 5" onclick="steal()"/></svg>`);
    const question = "<img src='https://evil.example/question' onerror='steal()'>";
    const report = previewReport(createAriaAnswerDocument(question, source));

    expect(report.getElementsByTagName("header")[0]).toHaveTextContent(question);
    expect(report.body).toHaveTextContent("Safe answer linked text");
    expect(report.body).not.toHaveTextContent("Secret action receipt");
    expect(report.body).not.toHaveTextContent("Copy answer");
    expect(report.body).not.toHaveTextContent("Hidden tooltip");
    expect(report.body).not.toHaveTextContent("Decoration");
    expect(report.body).not.toHaveTextContent("Injected HTML");
    for (const tag of ["script", "iframe", "img", "image", "foreignObject", "button"]) {
      expect(report.getElementsByTagName(tag)).toHaveLength(0);
    }
    expect(report.getElementsByTagName("a")[0]).not.toHaveAttribute("href");
    expect(report.getElementsByTagName("path")[0]).not.toHaveAttribute("onclick");
    expect(report.getElementsByTagName("rect")[0]).not.toHaveAttribute("fill");
    expect(report.head).not.toHaveTextContent("evil.example");
    expect(report.getElementsByTagName("meta")[0].content).toContain("default-src 'none'");
  });

  it("should retain local SVG clipping and gradients while omitting hover overlays", () => {
    const source =
      answer(`<svg viewBox="0 0 100 100"><defs><clipPath id="plot"><rect width="100" height="100"/></clipPath>
      <linearGradient id="series"><stop offset="0" stop-color="#123456"/></linearGradient></defs>
      <path d="M0 0 L10 10" clip-path="url(#plot)" fill="url(#series)"/>
      <rect class="recharts-tooltip-cursor" width="100" height="100"/></svg>`);
    const report = previewReport(createAriaAnswerDocument("Show spend", source));
    const path = report.getElementsByTagName("path")[0];
    expect(path).toHaveAttribute("clip-path", "url(#plot)");
    expect(path).toHaveAttribute("fill", "url(#series)");
    expect(report.getElementsByTagName("rect")).toHaveLength(1);
    expect(report.getElementsByTagName("stop")[0]).toHaveAttribute("stop-color", "#123456");
  });
});

function printFrame() {
  const frame = document.createElement("iframe");
  const printWindow = window;
  vi.spyOn(frame, "contentWindow", "get").mockReturnValue(printWindow);
  const print = vi.spyOn(printWindow, "print").mockImplementation(() => undefined);
  vi.spyOn(printWindow, "focus").mockImplementation(() => undefined);
  const createElement = document.createElement.bind(document);
  vi.spyOn(document, "createElement").mockImplementation((tagName, options) =>
    tagName === "iframe" ? frame : createElement(tagName, options),
  );
  return { frame, printWindow, print };
}

describe("exportAriaAnswer", () => {
  it("should print only the supplied answer in an isolated document and clean up after printing", () => {
    vi.useFakeTimers();
    const source = answer("<p>The answer is $3.25.</p><button>Export answer</button>");
    const { frame, printWindow, print } = printFrame();
    expect(exportAriaAnswer("How much did I spend?", source)).toBe(true);
    expect(print).toHaveBeenCalledOnce();
    expect(frame).toHaveAttribute("sandbox", "allow-modals allow-same-origin");
    expect(frame.contentDocument?.body).toHaveTextContent("How much did I spend?");
    expect(frame.contentDocument?.body).toHaveTextContent("The answer is $3.25.");
    expect(frame.contentDocument?.body).not.toHaveTextContent("Export answer");
    printWindow.dispatchEvent(new Event("afterprint"));
    expect(frame.isConnected).toBe(false);
    expect(source.isConnected).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("should report unavailable printing and remove the temporary document", () => {
    vi.useFakeTimers();
    const source = answer("<p>Answer</p>");
    const { frame, print } = printFrame();
    print.mockImplementation(() => {
      throw new Error("Printing unavailable");
    });
    expect(exportAriaAnswer("Question", source)).toBe(false);
    expect(frame.isConnected).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });
});
