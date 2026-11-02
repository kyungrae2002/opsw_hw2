const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"];

const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const dropzone = $("dropzone");
const preview = $("preview");
const previewImg = $("previewImg");
const previewName = $("previewName");
const analyzeBtn = $("analyze");
const statusEl = $("status");
const resultEl = $("result");

let selected = null;
let previewUrl = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function selectFile(file) {
  resultEl.hidden = true;
  if (!file) return;
  if (!ALLOWED.includes(file.type)) {
    selected = null;
    analyzeBtn.disabled = true;
    preview.hidden = true;
    setStatus("이미지 파일(JPG, PNG, WEBP, BMP, GIF)만 올릴 수 있어요.", true);
    return;
  }
  if (file.size > MAX_BYTES) {
    selected = null;
    analyzeBtn.disabled = true;
    preview.hidden = true;
    setStatus("5MB 이하 이미지만 올릴 수 있어요.", true);
    return;
  }
  selected = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImg.src = previewUrl;
  previewName.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  preview.hidden = false;
  analyzeBtn.disabled = false;
  setStatus("준비 완료. 분석하기를 눌러주세요.");
}

function renderResult(data) {
  const list = $("predictions");
  list.innerHTML = "";
  data.predictions.forEach((p) => {
    const pct = (p.probability * 100).toFixed(1);
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="row"><span class="label"></span><span class="pct">${pct}%</span></div>
      <div class="bar"><span style="width:${Math.max(p.probability * 100, 1)}%"></span></div>`;
    li.querySelector(".label").textContent = `${p.rank}. ${p.label}`;
    list.appendChild(li);
  });
  $("meta").textContent = `${data.model} · ${data.inference_ms} ms`;
  $("notice").textContent = data.notice;
  resultEl.hidden = false;
}

async function analyze() {
  if (!selected) return;
  analyzeBtn.disabled = true;
  setStatus("분석 중…");
  try {
    const form = new FormData();
    form.append("file", selected);
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `서버 오류 (${res.status})`);
    renderResult(data);
    setStatus("분석 완료");
  } catch (err) {
    resultEl.hidden = true;
    setStatus(err.message || "분석 중 오류가 발생했습니다.", true);
  } finally {
    analyzeBtn.disabled = !selected;
  }
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
analyzeBtn.addEventListener("click", analyze);

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    const model = h.model.loaded ? h.model.name : "모델 미준비";
    $("deploy").textContent = `v${h.version} · commit ${h.commit} · ${model}`;
  })
  .catch(() => { $("deploy").textContent = "서버 상태를 확인할 수 없습니다."; });
