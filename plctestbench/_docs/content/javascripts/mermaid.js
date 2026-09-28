document$.subscribe(() => {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({ startOnLoad: false, theme: "neutral" });
    mermaid.run({ querySelector: ".mermaid" });
  }
});
