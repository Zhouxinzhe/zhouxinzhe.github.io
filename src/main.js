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
}

function isActive(label) {
  return label.toLowerCase() === page || (label === "Home" && page === "home");
}

function renderHome() {
  const latestPosts = posts.slice(0, 5);
  shell(`
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Automation · Multi-agent Systems · Robotics</p>
        <h1>${profile.name}<br><span>${profile.chineseName}</span></h1>
        <p class="lead">
          Undergraduate student at Shanghai Jiao Tong University. I write about control,
          robotics, graph learning, and the technical ideas I am trying to understand well.
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
          <span>${profile.affiliation}</span>
        </div>
        <dl>
          ${profile.metrics.map((item) => `<div><dt>${item.label}</dt><dd>${item.value}</dd></div>`).join("")}
        </dl>
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
        My current interests lie around multi-agent systems, formation maneuver control,
        robotics, graph neural networks, and geometric representation learning.
      </p>
      <div class="hero-actions">
        <a href="mailto:${profile.email}">Email</a>
        <a href="${profile.github}">GitHub</a>
        <a href="https://arxiv.org/abs/2505.05795">arXiv</a>
      </div>
    </section>
    <section class="section split">
      <div>
        <p class="section-label">Education</p>
        <h2>${profile.title}</h2>
        <p>${profile.affiliation}</p>
        <div class="metric-row">
          ${profile.metrics.map((item) => `<div><span>${item.value}</span><small>${item.label}</small></div>`).join("")}
        </div>
      </div>
      <div>
        <p class="section-label">Research Interests</p>
        <div class="tag-list">${profile.interests.map((item) => `<span>${item}</span>`).join("")}</div>
      </div>
    </section>
    <section class="section">
      <p class="section-label">Publications & Manuscripts</p>
      <h2>Selected Work</h2>
      <div class="paper-list">${profile.publications.map(publicationCard).join("")}</div>
    </section>
    <section class="section split">
      <div>
        <p class="section-label">Experience</p>
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
      </div>
      <div>
        <p class="section-label">Honors</p>
        <h2>Selected Awards</h2>
        <ul class="clean-list">${profile.honors.map((honor) => `<li>${honor}</li>`).join("")}</ul>
      </div>
    </section>
  `);
}

function renderAbout() {
  shell(`
    <section class="page-hero">
      <p class="eyebrow">About</p>
      <h1>你好，我是${profile.chineseName}</h1>
      <p class="lead">
        上海交通大学自动化专业本科生。这个站点会继续保留学习笔记和技术博客，
        但整体表达会更偏学术主页：更清晰地呈现研究方向、论文、项目和联系方式。
      </p>
    </section>
    <section class="section split">
      <div>
        <p class="section-label">Profile</p>
        <h2>Education & Research</h2>
        <p>
          我目前在上海交通大学自动化与感知学院（School of Automation and Intelligent Sensing）就读自动化专业，研究兴趣主要集中在
          多智能体系统、编队控制、机器人、图神经网络与几何深度学习。
        </p>
        <p>
          核心学积分 94.678，绩点 4.12/4.3，专业排名 1/93。
        </p>
      </div>
      <div>
        <p class="section-label">Contact</p>
        <h2>Get in Touch</h2>
        <ul class="contact-list">
          <li><span>Email</span><a href="mailto:${profile.email}">${profile.email}</a></li>
          <li><span>Personal</span><a href="mailto:${profile.personalEmail}">${profile.personalEmail}</a></li>
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
