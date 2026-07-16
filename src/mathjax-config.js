window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["ams"] }
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  },
  startup: {
    typeset: false,
    ready() {
      MathJax.startup.defaultReady();
      window.dispatchEvent(new Event("mathjax-ready"));
    }
  }
};
