import re
import urllib.request

req = urllib.request.Request(
    "https://sites.google.com/view/ybenezeth/ubfcrppg",
    headers={"User-Agent": "Mozilla/5.0"},
)
html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
links = sorted(set(re.findall(r"https?://[^\"\\\s<>]+", html)))
keys = ("drive", "dropbox", "zip", "ubfc", "download", "bourgogne", "figshare", "zenodo", "onedrive", "github")
for url in links:
    low = url.lower()
    if any(key in low for key in keys):
        print(url[:400])
print("TOTAL", len(links))
