const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');

function extractText(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (item && typeof item === 'object') {
          const type = String(item.type || '').toLowerCase();
          if (type === 'thinking' || type === 'reasoning') return '';
          if (item.text !== undefined) return extractText(item.text);
          if (item.content !== undefined) return extractText(item.content);
          if (item.msg !== undefined) return extractText(item.msg);
        }
        return extractText(item);
      })
      .filter(Boolean)
      .join('\n');
  }
  if (typeof value === 'object') {
    const type = String(value.type || '').toLowerCase();
    if (type === 'thinking' || type === 'reasoning') return '';
    for (const key of ['answer', 'text', 'content', 'message', 'msg', 'detail']) {
      if (value[key] !== undefined && value[key] !== null) {
        return extractText(value[key]);
      }
    }
    try {
      return JSON.stringify(value);
    } catch {
      return '';
    }
  }
  return String(value);
}

function escapeHtml(s = '') {
  return String(s ?? '').replace(/[&<>'"]/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;',
  }[c]));
}

function formatText(s = '') {
  const normalized = extractText(s);
  return escapeHtml(normalized)
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function formatErrorDetail(detail, fallback = 'Request failed') {
  const text = extractText(detail).trim();
  return text || fallback;
}

async function parseApiResponse(res) {
  const raw = await res.text();
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { detail: raw };
  }
}

function addMessage(role, text, source = '', citations = []) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const citeList = Array.isArray(citations) ? citations : [];
  const citeHtml = citeList.length
    ? `<div class="citations"><strong>Sources</strong><br>${citeList
        .map((c) => {
          const item = c && typeof c === 'object' ? c : { title: c };
          const url = typeof item.url === 'string' && /^https?:\/\//i.test(item.url.trim()) ? item.url.trim() : '';
          const title = extractText(item.title || url) || 'Source';
          return url
            ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(title)}</a>`
            : escapeHtml(title);
        })
        .join('<br>')}</div>`
    : '';
  const safeSource = extractText(source);
  const avatarLabel = role === 'user' ? 'You' : 'AI';
  wrap.innerHTML = `<div class="avatar">${avatarLabel}</div><div class="bubble">${formatText(text)}${
    safeSource ? `<div class="answer-source">Source: ${escapeHtml(safeSource)}</div>` : ''
  }${citeHtml}</div>`;
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function renderTrace(items = []) {
  const list = Array.isArray(items) ? items : [];
  trace.innerHTML = list.length
    ? list.map((x) => `<div class="trace-item">${escapeHtml(extractText(x))}</div>`).join('')
    : '<div class="empty">No trace.</div>';
}

async function askAgent(q) {
  addMessage('user', q);
  question.value = '';
  renderTrace(['Running LangGraph workflow...']);
  sourceUsed.textContent = 'Running';
  const btn = form.querySelector('button');
  btn.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await parseApiResponse(res);
    if (!res.ok) {
      throw new Error(formatErrorDetail(data.detail, `Request failed (${res.status})`));
    }
    const answerText = extractText(data.answer) || 'No answer was returned.';
    const sourceText = extractText(data.source_used);
    addMessage('assistant', answerText, sourceText, data.citations || []);
    renderTrace(data.trace || []);
    sourceUsed.textContent = sourceText || '—';
  } catch (e) {
    addMessage('assistant', `Error: ${formatErrorDetail(e && e.message, 'Request failed')}`);
    renderTrace(['Request failed']);
    sourceUsed.textContent = 'Error';
  } finally {
    btn.disabled = false;
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = question.value.trim();
  if (q) askAgent(q);
});

document.querySelectorAll('.example').forEach((b) =>
  b.addEventListener('click', () => askAgent(b.textContent.trim()))
);

const modal = document.getElementById('uploadModal');
document.getElementById('openUpload').onclick = () => modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick = () => modal.classList.add('hidden');
document.getElementById('uploadBtn').onclick = async () => {
  const file = document.getElementById('fileInput').files[0];
  const key = document.getElementById('adminKey').value;
  const status = document.getElementById('uploadStatus');
  if (!file) {
    status.textContent = 'Choose a file first.';
    return;
  }
  status.textContent = 'Indexing document...';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'X-Admin-Key': key },
      body: fd,
    });
    const d = await parseApiResponse(r);
    if (!r.ok) {
      throw new Error(formatErrorDetail(d.detail, `Upload failed (${r.status})`));
    }
    status.textContent = `Indexed ${d.file}: ${d.chunks} chunks.`;
  } catch (e) {
    status.textContent = `Error: ${formatErrorDetail(e && e.message, 'Upload failed')}`;
  }
};
