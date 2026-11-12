# Diagrams

The architecture docs use hand-laid-out SVGs instead of Mermaid, so each diagram
fits a docs column without wasted space.

- Every SVG embeds [diagram-style.css](diagram-style.css) verbatim in one
  `<style>` element. The stylesheet is based on the litellm-docs theme (Geist
  and Geist Mono, the accent `#0117be` and its greys) and switches to the dark
  palette with `prefers-color-scheme`
- Use only its classes, with no inline styles or hard-coded colors. The box
  classes mean: `box core` the Rust router core, `box host` the host and Python
  side, `box store` caches and Redis, `box warn` errors, `box ok` results, and
  `box decision` a question
- Keep the `viewBox` at most 880 wide and as short as the content allows. Start
  with the `canvas` rect, and give the root element `role="img"` and a `<title>`
- Edit a diagram by changing its SVG directly. A page's diagrams live in an
  `img/` folder next to it
