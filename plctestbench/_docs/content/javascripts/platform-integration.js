(() => {
  const MESSAGE_TYPE = "plctestbench-theme";
  const EMBEDDED_CLASS = "plctestbench-embedded";
  const ALLOWED_THEMES = new Set(["light", "dark"]);

  function applyTheme(theme) {
    if (!ALLOWED_THEMES.has(theme) || !document.body) return;

    const scheme = theme === "dark" ? "slate" : "default";
    document.body.setAttribute("data-md-color-scheme", scheme);
    document.documentElement.style.colorScheme = theme;
  }

  if (window.parent === window) return;

  document.documentElement.classList.add(EMBEDDED_CLASS);
  window.addEventListener("message", (event) => {
    if (event.source !== window.parent || event.origin !== window.location.origin) return;
    if (event.data?.type !== MESSAGE_TYPE) return;
    applyTheme(event.data.theme);
  });
})();
