import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";

const root = path.resolve(process.argv[2] || ".");
const port = Number(process.argv[3] || 4173);

const types = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml"
};

createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://localhost:${port}`);
  const pathname = decodeURIComponent(url.pathname);
  const candidates = [
    path.join(root, pathname),
    path.join(root, pathname, "index.html"),
    path.join(root, "404.html")
  ];

  for (const candidate of candidates) {
    try {
      const info = await stat(candidate);
      if (!info.isFile()) continue;
      response.writeHead(candidate.endsWith("404.html") ? 404 : 200, {
        "content-type": types[path.extname(candidate)] || "application/octet-stream"
      });
      createReadStream(candidate).pipe(response);
      return;
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`Preview server running at http://localhost:${port}`);
  console.log(`Serving ${root}`);
});
