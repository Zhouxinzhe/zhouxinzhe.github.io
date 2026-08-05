export function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function formatDate(dateString) {
  if (!dateString) return "";
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return dateString;
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "2-digit"
  }).format(date);
}

export function postUrl(post) {
  return `./post.html?slug=${encodeURIComponent(post.slug)}`;
}

export function summarize(value = "", length = 150) {
  const text = value
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*]\([^)]+\)/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\\\[[\s\S]*?\\\]/g, " ")
    .replace(/\$[^$\n]+\$/g, " ")
    .replace(/\[([^\]]+)]\([^)]+\)/g, "$1")
    .replace(/[#>*_`$-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > length ? `${text.slice(0, length)}...` : text;
}

export function groupByYear(posts) {
  return posts.reduce((groups, post) => {
    const year = post.date ? post.date.slice(0, 4) : "Undated";
    if (!groups.has(year)) groups.set(year, []);
    groups.get(year).push(post);
    return groups;
  }, new Map());
}

export function markdownToHtml(markdown = "") {
  const blocks = [];
  let source = markdown.replace(/\r\n/g, "\n");

  source = source.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, language = "", code) => {
    const token = `@@CODE_${blocks.length}@@`;
    blocks.push(`<pre><code class="language-${escapeHtml(language)}">${escapeHtml(code.trim())}</code></pre>`);
    return token;
  });

  source = source.replace(/(^|\n)[ \t]*(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])(?=\s*(?:\n|$))/g, (_, prefix, formula) => {
    const token = `@@MATH_${blocks.length}@@`;
    blocks.push(`<div class="math-display">${escapeHtml(formula.trim())}</div>`);
    return `${prefix}${token}`;
  });

  const lines = source.split("\n");
  const html = [];
  let list = null;
  let listItemOpen = false;
  let paragraph = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const closeList = () => {
    if (!list) return;
    if (listItemOpen) {
      html.push("</li>");
      listItemOpen = false;
    }
    html.push(`</${list}>`);
    list = null;
  };

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const rawLine = lines[lineIndex];
    const line = rawLine.trim();
    if (!line) {
      const nextLine = lines
        .slice(lineIndex + 1)
        .map((candidate) => candidate.trim())
        .find((candidate) => candidate && !candidate.startsWith("@@MATH_")) ?? "";
      const continuesList =
        (list === "ol" && /^\d+\.\s+/.test(nextLine)) ||
        (list === "ul" && /^[-*]\s+/.test(nextLine));

      if (!continuesList) {
        flushParagraph();
        closeList();
      }
      continue;
    }

    if (line.startsWith("@@CODE_")) {
      flushParagraph();
      if (!listItemOpen) closeList();
      html.push(line);
      continue;
    }

    if (line.startsWith("@@MATH_")) {
      flushParagraph();
      if (!listItemOpen) closeList();
      html.push(line);
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length + 1;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const unordered = line.match(/^[-*]\s+(.+)$/);
    if (unordered) {
      flushParagraph();
      if (list !== "ul") {
        closeList();
        list = "ul";
        html.push("<ul>");
      }
      if (listItemOpen) html.push("</li>");
      html.push(`<li>${inlineMarkdown(unordered[1])}`);
      listItemOpen = true;
      continue;
    }

    const ordered = line.match(/^\d+\.\s+(.+)$/);
    if (ordered) {
      flushParagraph();
      if (list !== "ol") {
        closeList();
        list = "ol";
        html.push("<ol>");
      }
      if (listItemOpen) html.push("</li>");
      html.push(`<li>${inlineMarkdown(ordered[1])}`);
      listItemOpen = true;
      continue;
    }

    paragraph.push(line);
  }

  flushParagraph();
  closeList();

  return html.join("\n").replace(/@@(?:CODE|MATH)_(\d+)@@/g, (_, index) => blocks[Number(index)] ?? "");
}

function inlineMarkdown(value) {
  const images = [];
  const source = String(value).replace(/<img\b[^>]*>/gi, (tag) => {
    const src = tag.match(/\bsrc\s*=\s*["']([^"']+)["']/i)?.[1]?.trim();
    const alt = tag.match(/\balt\s*=\s*["']([^"']*)["']/i)?.[1]?.trim() || "";

    if (!src || !isSafeImageSource(src)) return "";

    const token = `@@INLINE_IMAGE_${images.length}@@`;
    images.push(`<img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy">`);
    return token;
  });

  return escapeHtml(source)
    .replace(/!\[([^\]]*)]\(([^)]+)\)/g, '<img src="$2" alt="$1">')
    .replace(/\[([^\]]+)]\(([^)]+)\)/g, '<a href="$2">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/@@INLINE_IMAGE_(\d+)@@/g, (_, index) => images[Number(index)] ?? "");
}

function isSafeImageSource(value) {
  return /^(https?:\/\/|\/(?!\/)|\.\.?\/)/i.test(value);
}
