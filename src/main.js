import { posts } from "./generated/posts.js";
import { profile } from "./data/profile.js";
import { escapeHtml, formatDate, groupByYear, markdownToHtml, postUrl, summarize } from "./utils.js";

const app = document.querySelector("#app");
const page = document.body.dataset.page || "home";

const navItems = [
  ["Home", "./index.html"],
  ["Academic", "./academic.html"],
  ["Archive", "./archive.html"],
  ["About", "./about.html"]
];

function shell(content) {
  app.innerHTML = `
    <header class="site-nav">
      <a class="brand" href="./index.html" aria-label="Home">
        <span>${profile.chineseName}</span>
        <small>${profile.name}</small>
      </a>
      <nav aria-label="Primary navigation">
        ${navItems
          .map(([label, href]) => `<a class="${isActive(label) ? "active" : ""}" href="${href}">${label}</a>`)
          .join("")}
      </nav>
    </header>
    <main>${content}</main>
    <footer class="site-footer">
      <span>${profile.name}</span>
      <span>${profile.affiliation}</span>
      <a href="mailto:${profile.email}">${profile.email}</a>
    </footer>
  `;
  typesetMath(app);
}

function typesetMath(root) {
  const render = () => {
    const mathJax = window.MathJax;
    if (!mathJax?.typesetPromise) return;

    mathJax.typesetClear([root]);
    mathJax.typesetPromise([root]).catch((error) => {
      console.error("MathJax rendering failed:", error);
    });
  };

  if (window.MathJax?.startup?.promise) {
    window.MathJax.startup.promise.then(render);
  } else {
    window.addEventListener("mathjax-ready", render, { once: true });
  }
}

function isActive(label) {
  return label.toLowerCase() === page || (label === "Home" && page === "home");
}

function renderHome() {
  const latestPosts = posts.slice(0, 5);
  shell(`
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Shanghai Jiao Tong University</p>
        <h1>${profile.name}<br><span>${profile.chineseName}</span></h1>
        <p class="lead">
          First-year graduate student in Control Science and Engineering at Shanghai Jiao Tong University.
          My current research focuses on offline reinforcement learning.
        </p>
        <div class="hero-actions">
          <a href="./academic.html">Academic Profile</a>
          <a href="./archive.html">Read Notes</a>
        </div>
      </div>
      <aside class="identity-panel" aria-label="Profile summary">
        <img src="${profile.avatar}" alt="${profile.name}">
        <div>
          <strong>${profile.title}</strong>
          <span class="identity-affiliation">${profile.affiliation.split(", ").map(escapeHtml).join("<br>")}</span>
        </div>
      </aside>
    </section>
    <section class="section focus-section">
      <div class="focus-intro">
        <p class="section-label">Research</p>
        <h2>Current Focus</h2>
        <div class="tag-list">
          ${profile.interests.map((interest) => `<span>${interest}</span>`).join("")}
        </div>
      </div>
      <div class="focus-papers">
        <div class="section-heading-row">
          <div>
            <p class="section-label">Selected Publications</p>
            <h2>Research Output</h2>
          </div>
          <a class="text-link" href="./academic.html">Academic profile</a>
        </div>
        <div class="paper-list compact">
        ${profile.publications.map(publicationCard).join("")}
        </div>
      </div>
    </section>
    <section class="section">
      <div class="section-heading-row">
        <div>
          <p class="section-label">Writing</p>
          <h2>Latest Notes</h2>
        </div>
        <a class="text-link" href="./archive.html">All posts</a>
      </div>
      <div class="post-grid">
        ${latestPosts.map(postCard).join("")}
      </div>
    </section>
  `);
}

function renderAcademic() {
  shell(`
    <section class="page-hero academic-hero">
      <p class="eyebrow">${profile.affiliation}</p>
      <h1>Academic Profile</h1>
      <p class="lead">
        I am a first-year graduate student in Control Science and Engineering, currently focusing on
        offline reinforcement learning. My previous work spans control, robotics, and geometric learning.
      </p>
      <div class="hero-actions">
        <a href="mailto:${profile.email}">Email</a>
        <a href="${profile.github}">GitHub</a>
      </div>
    </section>
    <section class="section research-overview">
      <div class="education-summary">
        <p class="section-label">Education</p>
        <h2>${profile.title}</h2>
        <p>${profile.affiliation}</p>
        <p>Undergraduate academic record</p>
        <div class="metric-row">
          ${profile.metrics.map((item) => `<div><span>${item.value}</span><small>${item.label}</small></div>`).join("")}
        </div>
      </div>
      <div class="research-interests">
        <p class="section-label">Research Interests</p>
        <h2>Offline Reinforcement Learning</h2>
        <div class="tag-list">${profile.interests.map((item) => `<span>${item}</span>`).join("")}</div>
      </div>
    </section>
    <section class="section">
      <p class="section-label">Publications & Manuscripts</p>
      <h2>Selected Work</h2>
      <div class="paper-list">${profile.publications.map(publicationCard).join("")}</div>
    </section>
    <section class="section timeline-section">
      <p class="section-label">Research Direction</p>
      <h2>Research Timeline</h2>
      <div class="timeline">
        ${profile.experience
          .map(
            (item) => `
              <article>
                <time>${item.time}</time>
                <h3>${item.title}</h3>
                <p>${item.description}</p>
              </article>`
          )
          .join("")}
      </div>
    </section>
    <section class="section honors-section">
      <p class="section-label">Honors</p>
      <h2>Selected Awards</h2>
      <ul class="clean-list two-column">${profile.honors.map((honor) => `<li>${honor}</li>`).join("")}</ul>
    </section>
  `);
}

function renderAbout() {
  shell(`
    <section class="page-hero">
      <p class="eyebrow">About</p>
      <h1>About ${profile.name}</h1>
      <p class="lead">
        I am a first-year graduate student in Control Science and Engineering at Shanghai Jiao Tong University.
        This website brings together my research profile, publications, and technical notes.
      </p>
    </section>
    <section class="section split">
      <div>
        <p class="section-label">Profile</p>
        <h2>Education & Research</h2>
        <p>
          I study at the School of Automation and Intelligent Sensing, Shanghai Jiao Tong University.
          My current research focuses on offline reinforcement learning. My previous work includes multi-agent
          control, robotics, graph neural networks, and geometric deep learning.
        </p>
        <p>
          During my undergraduate studies in Automation, I earned a core average of 94.678 and a GPA of 4.12/4.3,
          ranking 1st out of 93 students in my major.
        </p>
      </div>
      <div>
        <p class="section-label">Contact</p>
        <h2>Get in Touch</h2>
        <ul class="contact-list">
          <li><span>Email</span><a href="mailto:${profile.email}">${profile.email}</a></li>
          <li><span>WeChat</span><strong>zxz1150146797</strong></li>
          <li><span>GitHub</span><a href="${profile.github}">Zhouxinzhe</a></li>
        </ul>
      </div>
    </section>
    <section class="section">
      <p class="section-label">Honors</p>
      <h2>Selected Achievements</h2>
      <ul class="clean-list two-column">${profile.honors.map((honor) => `<li>${honor}</li>`).join("")}</ul>
    </section>
  `);
}

function renderArchive() {
  const tags = [...new Set(posts.flatMap((post) => post.tags))].sort((a, b) => a.localeCompare(b));
  const groups = groupByYear(posts);

  shell(`
    <section class="page-hero">
      <p class="eyebrow">Archive</p>
      <h1>Notes & Essays</h1>
      <p class="lead">A searchable collection of course notes, paper reading notes, and technical experiments.</p>
    </section>
    <section class="section archive-tools">
      <label>
        <span>Search</span>
        <input id="post-search" type="search" placeholder="Search posts, tags, or summaries">
      </label>
      <div class="tag-filter">
        <button class="active" data-tag="">All</button>
        ${tags.map((tag) => `<button data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`).join("")}
      </div>
    </section>
    <section class="section archive-list" id="archive-list">
      ${archiveMarkup(groups)}
    </section>
  `);

  wireArchiveSearch();
}

function renderPost() {
  const slug = new URLSearchParams(window.location.search).get("slug");
  const post = posts.find((item) => item.slug === slug) || posts[0];

  if (!post) {
    renderNotFound();
    return;
  }

  document.title = `${post.title} | ${profile.chineseName}`;
  shell(`
    <article class="post-page">
      <a class="text-link" href="./archive.html">Back to archive</a>
      <header>
        <p class="eyebrow">${formatDate(post.date)} · ${post.tags.join(" / ")}</p>
        <h1>${escapeHtml(post.title)}</h1>
        ${post.subtitle ? `<p class="lead">${escapeHtml(post.subtitle)}</p>` : ""}
      </header>
      <div class="markdown-body">${markdownToHtml(post.content)}</div>
    </article>
  `);
}

function renderNotFound() {
  shell(`
    <section class="page-hero">
      <p class="eyebrow">404</p>
      <h1>Page not found</h1>
      <p class="lead">The page may have moved during the migration to the new front-end site.</p>
      <div class="hero-actions">
        <a href="./index.html">Home</a>
        <a href="./archive.html">Archive</a>
      </div>
    </section>
  `);
}

function postCard(post) {
  return `
    <article class="post-card">
      <a href="${postUrl(post)}">
        <time>${formatDate(post.date)}</time>
        <h3>${escapeHtml(post.title)}</h3>
        <p>${escapeHtml(summarize(post.content, 120))}</p>
        <div>${post.tags.slice(0, 3).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div>
      </a>
    </article>
  `;
}

function publicationCard(item) {
  return `
    <article class="paper-card">
      <div class="paper-year">${item.year}</div>
      <div>
        <div class="paper-status">${item.status}</div>
        <h3>${item.title}</h3>
        <p class="muted">${item.authors}</p>
        <p class="muted">${item.venue}</p>
        <p>${item.summary}</p>
        ${
          item.links.length
            ? `<div class="inline-links">${item.links.map((link) => `<a href="${link.url}">${link.label}</a>`).join("")}</div>`
            : ""
        }
      </div>
    </article>
  `;
}

function archiveMarkup(groups) {
  return [...groups.entries()]
    .map(
      ([year, items]) => `
        <div class="archive-year" data-year="${year}">
          <h2>${year}</h2>
          <div class="archive-items">
            ${items.map(postCard).join("")}
          </div>
        </div>`
    )
    .join("");
}

function wireArchiveSearch() {
  const input = document.querySelector("#post-search");
  const buttons = [...document.querySelectorAll("[data-tag]")];
  let tag = "";

  const apply = () => {
    const query = input.value.trim().toLowerCase();
    const filtered = posts.filter((post) => {
      const haystack = `${post.title} ${post.subtitle} ${post.tags.join(" ")} ${post.content}`.toLowerCase();
      return (!tag || post.tags.includes(tag)) && (!query || haystack.includes(query));
    });
    document.querySelector("#archive-list").innerHTML = archiveMarkup(groupByYear(filtered));
  };

  input.addEventListener("input", apply);
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      tag = button.dataset.tag;
      apply();
    });
  });
}

const renderers = {
  home: renderHome,
  academic: renderAcademic,
  about: renderAbout,
  archive: renderArchive,
  post: renderPost,
  "not-found": renderNotFound
};

(renderers[page] || renderNotFound)();
