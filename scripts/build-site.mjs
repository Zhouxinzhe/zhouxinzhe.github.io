import { cp, mkdir, readdir, rm } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const dist = path.join(root, "dist");

const rootFiles = [
  "index.html",
  "about.html",
  "academic.html",
  "archive.html",
  "post.html",
  "404.html",
  ".nojekyll",
  "CNAME",
  "ads.txt",
  "sw.js"
];

const assetDirs = ["src", "img", "fonts", "pwa"];

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

for (const file of rootFiles) {
  await copyIfExists(path.join(root, file), path.join(dist, file));
}

for (const dir of assetDirs) {
  await copyIfExists(path.join(root, dir), path.join(dist, dir));
}

console.log(`Built static site in ${path.relative(root, dist)}/`);

async function copyIfExists(from, to) {
  try {
    await cp(from, to, { recursive: true });
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}
