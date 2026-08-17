import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.3/full/pyodide.mjs";

const INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.3/full/";
// __init__ が import する順。cli / __main__ はブラウザでは不要。
const MUT_FILES = ["__init__", "rs", "bits", "palette", "layout", "pose",
                   "codec", "render", "detect"];

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
let py, pyEncode, pyCheck, pyDecode, pyReset;

// --- 初期化 -----------------------------------------------------------------

async function boot() {
  try {
    setStatus("Pyodide 読み込み中…", "booting");
    py = await loadPyodide({ indexURL: INDEX_URL });
    setStatus("numpy / pillow 読み込み中…", "booting");
    await py.loadPackage(["numpy", "pillow"]);
    setStatus("mutsume 読み込み中…", "booting");
    await loadMutsume();
    pyEncode = py.globals.get("do_encode");
    pyCheck = py.globals.get("check_encode");
    pyDecode = py.globals.get("decode_rgba");
    pyReset = py.globals.get("reset_hints");
    setStatus("準備完了", "ready");
    enableUI();
    runCheck();
    syncEcc();
  } catch (e) {
    console.error(e);
    setStatus("初期化に失敗: " + e.message, "error");
  }
}

async function loadMutsume() {
  py.FS.mkdir("/mutsume");
  await Promise.all(MUT_FILES.map(async (f) => {
    const r = await fetch(`mutsume/${f}.py`, { cache: "no-store" });
    if (!r.ok) throw new Error(`mutsume/${f}.py (${r.status})`);
    py.FS.writeFile(`/mutsume/${f}.py`, await r.text());
  }));
  py.runPython("import sys; sys.path.insert(0, '/')");
  const helper = await (await fetch("mutsume_web.py", { cache: "no-store" })).text();
  py.runPython(helper);
}

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = "status " + (cls || "");
}

function enableUI() {
  $("enc-run").disabled = false;
  $("img-file").disabled = false;
  $("cam-start").disabled = false;
}

// --- タブ --------------------------------------------------------------------

document.querySelectorAll(".tabs button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $(b.dataset.tab).classList.add("active");
    if (b.dataset.tab !== "camera") stopCamera();
  });
});

// --- 生成 --------------------------------------------------------------------

$("enc-run").addEventListener("click", () => {
  const text = $("enc-text").value;
  if (!text) return;
  const r = JSON.parse(pyEncode(
    text, $("enc-ecc").value, $("enc-profile").value,
    parseInt($("enc-size").value, 10),
    $("enc-dark").value, $("enc-light").value,
    $("enc-grid").checked, $("enc-invert").checked,
    $("enc-dark-grid").checked));
  if (!r.ok) { showWarn(r.error); return; }
  hideWarn();
  document.querySelector("#encode .result").hidden = false;
  $("enc-img").src = "data:image/png;base64," + r.png;
  $("enc-img").hidden = false;
  $("enc-info").textContent =
    `profile=${r.profile}  palette=${r.palette}  R=${r.radius}`;
  const dl = $("enc-dl");
  dl.href = "data:image/png;base64," + r.png;
  dl.hidden = false;
  const dlsvg = $("enc-dl-svg");
  dlsvg.href = "data:image/svg+xml;base64," + r.svg;
  dlsvg.hidden = false;
});

// テキスト・設定に応じて生成可否をライブ判定し、不可なら注意文言を出す
let _checkTimer;
function scheduleCheck() {
  clearTimeout(_checkTimer);
  _checkTimer = setTimeout(runCheck, 150);
}
function runCheck() {
  if (!pyCheck) return;
  const r = JSON.parse(pyCheck(
    $("enc-text").value, $("enc-ecc").value, $("enc-profile").value));
  if (r.ok) { hideWarn(); $("enc-run").disabled = false; }
  else { showWarn(r.error); $("enc-run").disabled = true; }
}
function showWarn(msg) {
  const w = $("enc-warn");
  w.textContent = "⚠ " + msg;
  w.hidden = false;
}
function hideWarn() { $("enc-warn").hidden = true; }

$("enc-text").addEventListener("input", scheduleCheck);
["enc-ecc", "enc-profile"].forEach(
  (id) => $(id).addEventListener("change", runCheck));

function syncEcc() {
  // micro は ECC 固定なので選べないようにする
  $("enc-ecc").disabled = $("enc-profile").value === "micro";
}
$("enc-profile").addEventListener("change", syncEcc);

// --- 共通デコード ------------------------------------------------------------

function decodeImageData(imgData, opts = {}) {
  const { maxSymbols = 4, useHints = false, hintsOnly = false } = opts;
  const bytes = new Uint8Array(imgData.data.buffer.slice(0));
  const pybuf = py.toPy(bytes);
  try {
    const out = pyDecode(pybuf, imgData.width, imgData.height,
                         maxSymbols, useHints, hintsOnly);
    return JSON.parse(out);
  } finally {
    pybuf.destroy();
  }
}

function drawOverlay(ctx, results) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#3cdc3c";
  ctx.fillStyle = "#3cdc3c";
  ctx.font = "16px system-ui, sans-serif";
  for (const g of results) {
    if (!g.outline) continue;
    // 外形
    ctx.beginPath();
    g.outline.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    // ファインダ
    for (const [x, y] of g.finders) {
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.stroke();
    }
    // 向き: 中心 -> コーナー0
    const [cx, cy] = g.center, [tx, ty] = g.corner0;
    const ex = cx + (tx - cx) * 0.8, ey = cy + (ty - cy) * 0.8;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    const a = Math.atan2(ey - cy, ex - cx);
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - 10 * Math.cos(a - 0.4), ey - 10 * Math.sin(a - 0.4));
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - 10 * Math.cos(a + 0.4), ey - 10 * Math.sin(a + 0.4));
    ctx.stroke();
    // ラベル
    const minx = Math.min(...g.outline.map((p) => p[0]));
    const miny = Math.min(...g.outline.map((p) => p[1]));
    const label = g.text.length > 40 ? g.text.slice(0, 39) + "…" : g.text;
    ctx.fillText(label, minx, Math.max(16, miny - 6));
  }
}

// --- 画像から読み取り --------------------------------------------------------

async function decodeFile(file) {
  document.querySelector("#image .result").hidden = false;
  // SVG は createImageBitmap が不安定なので img 要素経由でラスタライズする。
  const isSvg = file.type === "image/svg+xml"
    || file.name.toLowerCase().endsWith(".svg");
  const url = URL.createObjectURL(file);
  const img = new Image();
  try {
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = () => rej(new Error("open failed"));
      img.src = url;
    });
  } catch {
    URL.revokeObjectURL(url);
    $("img-info").textContent = "画像を開けませんでした";
    return;
  }
  const w0 = img.naturalWidth || 512;
  const h0 = img.naturalHeight || 512;
  // SVG はベクタなので、小さければ復号精度のため拡大してラスタライズする
  const scale = isSvg
    ? Math.min(1600, Math.max(600, Math.max(w0, h0))) / Math.max(w0, h0)
    : Math.min(1, 1200 / w0);
  const canvas = $("img-canvas");
  canvas.width = Math.round(w0 * scale);
  canvas.height = Math.round(h0 * scale);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";  // SVG の透過部分に備えて白で塗る
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const t = performance.now();
  const results = decodeImageData(imgData, { maxSymbols: 4 });
  const dt = Math.round(performance.now() - t);
  drawOverlay(ctx, results);
  $("img-info").textContent = results.length
    ? results.map((g) => `${g.text}  (${g.profile}/${g.palette} R=${g.radius})`).join("\n")
      + `\n${results.length} 個 / ${dt} ms`
    : `見つかりませんでした (${dt} ms)`;
}

$("img-file").addEventListener("change", (e) => {
  if (e.target.files[0]) decodeFile(e.target.files[0]);
});

const dropZone = document.getElementById("img-drop");
["dragover", "dragenter"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("drag"); }));
dropZone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f && py) decodeFile(f);
});

// --- カメラ ------------------------------------------------------------------

let stream = null, camRunning = false, frameNo = 0;

$("cam-start").addEventListener("click", startCamera);
$("cam-stop").addEventListener("click", stopCamera);

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" } });
    const video = $("cam-video");
    video.srcObject = stream;
    await video.play();
    document.querySelector("#camera .result").hidden = false;
    pyReset();
    frameNo = 0;
    camRunning = true;
    $("cam-start").disabled = true;
    $("cam-stop").disabled = false;
    loopCamera();
  } catch (e) {
    $("cam-stat").textContent = "カメラを開けません: " + e.message;
  }
}

function stopCamera() {
  camRunning = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  $("cam-start").disabled = false;
  $("cam-stop").disabled = true;
}

function loopCamera() {
  if (!camRunning) return;
  const video = $("cam-video");
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) { setTimeout(loopCamera, 50); return; }
  const scale = Math.min(1, 640 / vw);
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  const canvas = $("cam-canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, w, h);
  const imgData = ctx.getImageData(0, 0, w, h);

  frameNo++;
  const full = frameNo % 10 === 1;   // 10 フレームに 1 回は全探索
  const t = performance.now();
  const results = decodeImageData(imgData,
    { maxSymbols: 2, useHints: true, hintsOnly: !full });
  const dt = Math.round(performance.now() - t);

  drawOverlay(ctx, results);
  $("cam-stat").textContent =
    `decode ${dt} ms   detected ${results.length}${full ? "  (full)" : ""}`;

  // 同期デコードでメインスレッドを塞ぐので、setTimeout で UI に息継ぎさせる
  setTimeout(loopCamera, 0);
}

boot();
