// Cloudflare Worker: resolves a short-link URL to its real cardboard/cfg?p=... target,
// server-side (not subject to browser CORS the way this page's own fetch() is), for viewer
// QR codes that encode a shortener link instead of the config URL directly.
//
// Deployed at https://cardboard-resolver.zickler.workers.dev/ (Workers & Pages -> paste this
// file's contents into the online editor -> Deploy). Not built/executed by anything in this
// repo -- this file is kept here purely as the source of truth for what's deployed, since
// Cloudflare's dashboard editor doesn't pull from git. Referenced from
// viewer/cardboard-profile.js's SHORT_LINK_RESOLVER_URL.
//
// Some redirect chains (notably Google's own goo.gl -> cardboard/cfg -> arvr.google.com
// chain) have an INTERMEDIATE hop that still carries the query string, then a final hop that
// strips it -- so this returns the last hop in the chain that still has a non-empty query
// string ("bestUrl"), not necessarily the literal final URL.

const ALLOWED_ORIGIN = "https://zickler.github.io";

// Only resolve known URL-shortener domains, so this can't be used as a generic
// "fetch any URL for me" open proxy.
const ALLOWED_HOST_SUFFIXES = [
  "goo.gl",
  "bit.ly",
  "tinyurl.com",
  "t.co",
  "ow.ly",
  "is.gd",
  "buff.ly",
  "rebrand.ly",
];

function isAllowedHost(hostname) {
  return ALLOWED_HOST_SUFFIXES.some((suffix) => hostname === suffix || hostname.endsWith(`.${suffix}`));
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    Vary: "Origin",
  };
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders() },
  });
}

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders() });
    }

    const target = new URL(request.url).searchParams.get("url");
    if (!target) return json({ error: "missing url param" }, 400);

    let targetUrl;
    try {
      targetUrl = new URL(target);
    } catch {
      return json({ error: "invalid url" }, 400);
    }
    if (!isAllowedHost(targetUrl.hostname)) {
      return json({ error: `host not allowed: ${targetUrl.hostname}` }, 403);
    }

    try {
      let current = targetUrl.toString();
      const hops = [current];
      for (let i = 0; i < 10; i++) {
        const res = await fetch(current, { redirect: "manual" });
        const location = res.headers.get("location");
        if (!location) break;
        current = new URL(location, current).toString();
        hops.push(current);
      }
      const bestUrl = [...hops].reverse().find((h) => new URL(h).search.length > 1) || current;
      return json({ finalUrl: current, bestUrl, hops });
    } catch (err) {
      return json({ error: String(err) }, 502);
    }
  },
};
