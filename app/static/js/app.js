// Frontend logic for the Security Requirements Assistant single-page UI.
//
// There's no build step or framework here - just plain functions attached to
// global scope, wired up via onclick="..." attributes in index.html. State
// lives in a handful of top-level `let` variables (currentSessionId,
// currentMode, currentProcessState, currentArtifactFilter). Everything talks
// to the backend via fetch() calls to the routes defined in app/api/.
//
// Rough map of what's below, in the order it appears:
//   - Process step ("Asset/Threat/Goal/Req") and mode ("elicitation/audit") switching, Finalize/Export actions
//   - Toast notifications
//   - Session sidebar (create/list/switch/rename/delete)
//   - Chat: sending messages, rendering bot/user messages, feedback buttons, option chips
//   - Document sidebar (upload, list, delete, drag & drop)
//   - Artifact dashboard (pending/approved cards, accept/refine/discard)

let currentSessionId = null;
let currentMode = 'elicitation'; // Default mode - matches the visually active mode button
let currentProcessState = 'Asset';

// --- THEME TOGGLE ---
// The <head> inline script in index.html already stamped data-theme on <html> (if a
// preference was stored) before this file loaded, to avoid a flash of the wrong theme.
// This just keeps the toggle icon in sync and handles clicks.
function getActiveTheme() {
    const stamped = document.documentElement.getAttribute('data-theme');
    if (stamped) return stamped;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function updateThemeIcon() {
    const icon = document.getElementById('themeToggleIcon');
    icon.className = getActiveTheme() === 'light' ? 'fas fa-sun' : 'fas fa-moon';
}

function toggleTheme() {
    const next = getActiveTheme() === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) { /* ignore - theme just won't persist */ }
    updateThemeIcon();
}

updateThemeIcon();

function setProcessState(state) {
    currentProcessState = state;

    // Reset all steps
    document.querySelectorAll('.step').forEach(el => el.classList.remove('active'));

    // Activate selected step
    document.getElementById(`step-${state}`).classList.add('active');
    document.getElementById('finalizeStepLabel').textContent = state;

    // Small Bot Message to indicate context switch
    appendMessage('bot', `*Switched context to: **${state} Modeling***. What would you like to discuss?`);
}

async function finalizeStep() {
    if(!currentSessionId) return;
    const btn = document.getElementById('finalizeBtn');
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Summarizing...';

    const formData = new FormData();
    formData.append('session_id', currentSessionId);
    formData.append('process_state', currentProcessState);

    try {
        const res = await fetch('/chat/finalize', { method: 'POST', body: formData });
        const data = await res.json();
        appendMessage('bot', data.summary, currentProcessState);
    } catch(e) {
        showToast('Finalize failed.', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = original;
    }
}

function exportArtifacts() {
    if(!currentSessionId) return;
    window.location.href = `/sessions/${currentSessionId}/export`;
}

// Function for switching the mode in UI
function setMode(mode) {
    currentMode = mode;
    document.getElementById('mode-audit').classList.remove('active');
    document.getElementById('mode-elicitation').classList.remove('active');
    document.getElementById(`mode-${mode}`).classList.add('active');

    const input = document.getElementById('userInput');
    if(mode === 'audit') {
        input.placeholder = "Enter requirement to audit (e.g., 'System must be secure')...";
    } else {
        input.placeholder = "Enter an idea or topic (e.g., 'I need a secure login')...";
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = 'fa-info-circle';
    if (type === 'success') icon = 'fa-check-circle';
    if (type === 'error') icon = 'fa-exclamation-circle';

    toast.innerHTML = `<i class="fas ${icon}"></i> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// --- INITIALIZATION ---
async function init() {
    await loadSessions();
    // If no session exists, create one automatically
    if(!currentSessionId) await createNewSession();
}
init();

// --- SESSION LOGIC ---
async function loadSessions() {
    const res = await fetch('/sessions');
    const sessions = await res.json();
    const list = document.getElementById('sessionList');
    list.innerHTML = '';

    sessions.forEach(s => {
        const div = document.createElement('div');
        div.className = `session-item ${s.id === currentSessionId ? 'active' : ''}`;
        div.innerHTML = `
            <span onclick="switchSession('${s.id}')" style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${s.title}">
                <i class="far fa-comments"></i> ${s.title}
            </span>
            <div class="session-actions" style="display: flex; gap: 8px;">
                <i class="fas fa-pen edit-btn" onclick="renameSession('${s.id}', '${s.title.replace(/'/g, "\\'")}', event)" style="opacity:0; cursor:pointer; font-size:0.85em; transition:0.2s;" title="Rename Chat"></i>
                <i class="fas fa-trash delete-btn" onclick="deleteSession('${s.id}')" style="opacity:0; transition:0.2s;"></i>
            </div>
        `;
        div.onmouseenter = () => {
            div.querySelector('.edit-btn').style.opacity = '0.6';
            div.querySelector('.delete-btn').style.opacity = '1';
        };
        div.onmouseleave = () => {
            div.querySelector('.edit-btn').style.opacity = '0';
            div.querySelector('.delete-btn').style.opacity = '0';
        };

        div.querySelector('.edit-btn').onmouseenter = (e) => e.target.style.opacity = '1';
        div.querySelector('.edit-btn').onmouseleave = (e) => e.target.style.opacity = '0.6';

        list.appendChild(div);
    });

    // Auto-select first if none selected
    if(sessions.length > 0 && !currentSessionId) {
        switchSession(sessions[0].id);
    }
}

async function createNewSession() {
    const res = await fetch('/sessions', { method: 'POST' });
    const s = await res.json();
    currentSessionId = s.id;
    await loadSessions();
    await switchSession(s.id);
}

async function switchSession(id) {
    currentSessionId = id;

    // Visual Update
    document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
    // Re-load sessions to update the "active" class rendered in loadSessions (or manually toggle class here)
    loadSessions();

    await loadHistory(id);
    await loadDocuments(id);
    await loadArtifacts(id);
}

async function deleteSession(id) {
    if(!confirm("Delete this chat and all its PDFs?")) return;
    await fetch(`/sessions/${id}`, { method: 'DELETE' });

    if(currentSessionId === id) currentSessionId = null;
    await loadSessions();

    // If we deleted the active one and list is empty, clear UI
    if(!currentSessionId) {
        document.getElementById('chatHistory').innerHTML = '';
        document.getElementById('docList').innerHTML = '';
        await createNewSession();
    }
}

async function renameSession(id, oldTitle, event) {
    event.stopPropagation();

    const newTitle = prompt("Update the chat name:", oldTitle);

    if (newTitle === null || newTitle.trim() === "") return;

    const formData = new FormData();
    formData.append('title', newTitle.trim());

    try {
        const res = await fetch(`/sessions/${id}/rename`, { method: 'POST', body: formData });
        if (res.ok) {
            await loadSessions();
            showToast("Chat successfully renamed!", "success");
        }
    } catch(e) {
        showToast("Error renaming the chat", "error");
    }
}

// --- CHAT LOGIC ---
async function loadHistory(id) {
    const res = await fetch(`/sessions/${id}/history`);
    const msgs = await res.json();
    const box = document.getElementById('chatHistory');
    box.innerHTML = '';

    if (msgs.length === 0) {
        box.innerHTML = `
            <div id="emptyState" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); text-align: center; animation: fadeIn 0.5s ease;">
                <div style="background: var(--accent-soft); padding: 20px; border-radius: 50%; margin-bottom: 20px;">
                    <i class="fas fa-robot" style="font-size: 3rem; color: var(--accent);"></i>
                </div>
                <h2 style="color: var(--text-primary); margin-bottom: 10px;">Welcome to the Security Architect!</h2>
                <p style="max-width: 450px; line-height: 1.6;">
                    I help you define ISO 29148-compliant security requirements.<br><br>
                    Upload a <strong>system document (PDF)</strong> on the right or select the process <strong>'1. Assets'</strong> to start interactively!
                </p>
            </div>
        `;
    } else {
        msgs.forEach(m => appendMessage(m.role, m.content, m.process_state));
    }
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if(!text) return;

    const existingMsgs = document.querySelectorAll('.message').length;
    const sendBtn = document.getElementById('sendBtn');

    input.disabled = true;
    sendBtn.disabled = true;
    document.getElementById('loadingLabel').textContent = 'Auditor is analyzing...';
    document.getElementById('loading').style.display = 'flex';

    // Optimistic UI update
    appendMessage('user', text);
    input.value = '';
    input.style.height = 'auto';

    const formData = new FormData();
    formData.append('session_id', currentSessionId);
    formData.append('user_message', text);
    formData.append('intent', currentMode);
    formData.append('process_state', currentProcessState);

    try {
        const res = await fetch('/chat', { method: 'POST', body: formData });
        const data = await res.json();

        let botText = data.answer;
        if(data.sources && data.sources.length > 0) {
            botText += `\n\n*(Sources: ${data.sources.join(', ')})*`;
        }
        appendMessage('bot', botText, currentProcessState);

        // If this was the first message, refresh session list to show new title
        if(existingMsgs === 0) {
            loadSessions();
        }

    } catch(e) {
        appendMessage('bot', '**Error:** Could not reach server.');
    } finally {
        document.getElementById('loading').style.display = 'none';
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

const STEP_ICONS = { 'Asset': 'fa-shield-alt', 'Threat': 'fa-biohazard', 'Goal': 'fa-bullseye', 'Req': 'fa-check' };

// Pulls "A. ...", "B. ...", "C. ..." option lines out of a bot reply so they can
// be rendered as clickable chips in addition to the plain markdown text.
function extractOptionChips(text) {
    // The model formats options as "A." or "A)" fairly interchangeably despite the
    // prompt not mandating either - matching both is what actually makes this fire
    // on real replies (an "A." only pattern missed most of them in practice).
    const matches = [...text.matchAll(/^[A-C][.)]\s+(.+)$/gm)];
    return matches.slice(0, 3).map(m => m[1].trim()).filter(Boolean);
}

function pickOption(encodedText) {
    const input = document.getElementById('userInput');
    input.value = decodeURIComponent(encodedText);
    autoResize(input);
    input.focus();
}

// Pulls the "- 🔴/🟡/🟢 **Criterion:** X/10 - reason" lines out of an Audit Report
// reply (see AUDIT_PROMPT's STRICT OUTPUT TEMPLATE in app/core/prompts.py) so they
// can be rendered as scannable score meters instead of only as plain list text.
function extractAuditScores(text) {
    const regex = /-\s*(?:🔴|🟡|🟢)\s*\*\*(.+?):\*\*\s*(\d{1,2})\s*\/\s*10/g;
    const scores = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
        const value = Math.max(0, Math.min(10, parseInt(match[2], 10)));
        scores.push({ label: match[1].trim(), value });
    }
    return scores;
}

function renderAuditScoresHtml(scores) {
    const rows = scores.map(s => {
        const tier = s.value <= 3 ? 'bad' : (s.value <= 7 ? 'warn' : 'good');
        return `
            <div class="audit-score-row">
                <span class="audit-score-label">${escapeHtml(s.label)}</span>
                <span class="audit-score-meter"><span class="audit-score-fill ${tier}" style="width:${s.value * 10}%"></span></span>
                <span class="audit-score-value">${s.value}/10</span>
            </div>
        `;
    }).join('');
    return `<div class="audit-scores">${rows}</div>`;
}

function appendMessage(role, text, processState) {
    const box = document.getElementById('chatHistory');

    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.remove();

    const div = document.createElement('div');
    div.className = 'message';

    let feedbackHtml = '';
    let chipsHtml = '';
    let scoresHtml = '';
    if(role === 'bot') {
        const scores = extractAuditScores(text);
        if (scores.length > 0) {
            scoresHtml = renderAuditScoresHtml(scores);
        }
        // Only the artifact type matching the step this message belongs to can be saved here -
        // falls back to the currently active step for older messages saved before this was tracked per-message.
        const stepType = processState || currentProcessState;
        const icon = STEP_ICONS[stepType] || 'fa-shield-alt';
        const safeText = encodeURIComponent(text).replace(/'/g, "%27");
        feedbackHtml = `
            <div style="margin-top: 10px; display: flex; gap: 5px; flex-wrap: wrap;">
                <button class="f-btn" onclick="saveArtifact('${stepType}', '${safeText}')"><i class="fas ${icon}"></i> Save as ${stepType}</button>
                <button class="f-btn" onclick="sendFeedback(this, 'up', '${safeText}')" title="Good response"><i class="fas fa-thumbs-up"></i></button>
                <button class="f-btn" onclick="sendFeedback(this, 'down', '${safeText}')" title="Bad response"><i class="fas fa-thumbs-down"></i></button>
            </div>
        `;

        const options = extractOptionChips(text);
        if(options.length > 0) {
            chipsHtml = `<div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px;">` +
                options.map(opt => `<button class="option-chip" onclick="pickOption('${encodeURIComponent(opt).replace(/'/g, "%27")}')">${escapeHtml(opt)}</button>`).join('') +
                `</div>`;
        }
    }

    div.innerHTML = `
        <div class="avatar ${role}">
            <i class="fas fa-${role==='user'?'user':'robot'}"></i>
        </div>
        <div class="msg-content">
            ${scoresHtml}
            ${marked.parse(text)}
            ${chipsHtml}
            ${feedbackHtml}
        </div>
    `;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

async function sendFeedback(btn, rating, encodedText) {
    const container = btn.parentElement;
    container.querySelectorAll('.f-btn').forEach(b => b.className = 'f-btn'); // reset
    btn.classList.add(rating === 'up' ? 'active-up' : 'active-down');

    const formData = new FormData();
    formData.append('session_id', currentSessionId);
    formData.append('bot_message', decodeURIComponent(encodedText));
    formData.append('rating', rating);

    try {
        await fetch('/feedback', { method: 'POST', body: formData });
    } catch(e) {
        console.error("Could not send feedback", e);
    }
}

function handleEnter(e) {
    if(e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// --- DOCUMENT LOGIC ---
async function loadDocuments(id) {
    const res = await fetch(`/sessions/${id}/documents`);
    const data = await res.json();
    const list = document.getElementById('docList');

    list.innerHTML = '';

    if(data.documents.length === 0) {
        list.innerHTML = '<div style="padding:10px; color:var(--text-muted); font-size:0.8em; text-align:center;">No user docs.<br>Supervisor knowledge active.</div>';
        return;
    }

    data.documents.forEach(doc => {
        const displayName = doc.includes('_') ? doc.split('_').slice(1).join('_') : doc;

        const div = document.createElement('div');
        div.className = 'doc-item';
        div.innerHTML = `
            <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 150px;" title="${doc}">${displayName}</span>
            <i class="fas fa-times" onclick="deleteDocument('${doc}')"></i>
        `;
        list.appendChild(div);
    });
}

// Not currently wired to a button in the UI (documents are auto-scanned on upload
// via the backend's background task) - kept for a manual re-scan action if one
// gets added to the UI later.
async function scanDocument(filename) {
    document.getElementById('loadingLabel').textContent = `Scanning ${filename} for artifacts...`;
    document.getElementById('loading').style.display = 'flex';

    try {
        const res = await fetch(`/sessions/${currentSessionId}/scan/${filename}`, { method: 'POST' });
        const data = await res.json();

        showToast(`Scan complete! Found and imported ${data.found_and_saved} artifacts.`, "success");

        await loadArtifacts(currentSessionId);
    } catch(e) {
        showToast("Scan failed.", "error");
    }

    document.getElementById('loading').style.display = 'none';
    document.getElementById('loadingLabel').textContent = 'Auditor is analyzing...';
}

async function uploadFile() {
    const input = document.getElementById('fileInput');
    if(input.files.length === 0) return;

    const btn = document.querySelector('.upload-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    btn.disabled = true;

    const formData = new FormData();
    formData.append('session_id', currentSessionId);
    formData.append('file', input.files[0]);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();

        input.value = '';

        await loadDocuments(currentSessionId);

        appendMessage('bot', `✅ **Knowledge basis updated:** The file \`${data.filename}\` has been successfully uploaded and is now available as background knowledge.\n\n🔍 **AI Analysis:** I am scanning the document in the background for artifacts. Your dashboard will update automatically when I find something.`);

        startArtifactPolling();

    } catch(e) {
        showToast("Upload failed.", "error");
    }

    btn.innerHTML = originalText;
    btn.disabled = false;
}

// Polls the artifact list every 5s for up to 2 minutes after an upload, since the
// backend's auto-scan runs in a background task with no push notification back to the UI.
function startArtifactPolling() {
    let attempts = 0;
    const interval = setInterval(async () => {
        attempts++;
        await loadArtifacts(currentSessionId);
        if(attempts >= 24) clearInterval(interval);
    }, 5000);
}

async function deleteDocument(filename) {
    if(!confirm(`Remove ${filename} from this chat?`)) return;
    await fetch(`/sessions/${currentSessionId}/documents/${filename}`, { method: 'DELETE' });
    await loadDocuments(currentSessionId);
}

async function saveArtifact(type, encodedContent) {
    let content = decodeURIComponent(encodedContent);

    // Strip markdown/HTML formatting only - no heuristic extraction. The saved
    // artifact lands as "pending" and can be refined in the Dashboard before acceptance.
    content = content.replace(/[#*`]/g, '');
    content = content.replace(/<[^>]*>?/gm, '');
    content = content.trim();

    if(content.length > 250) content = content.substring(0, 250) + '...';

    const formData = new FormData();
    formData.append('session_id', currentSessionId);
    formData.append('artifact_type', type);
    formData.append('content', content);

    try {
        const res = await fetch('/artifacts', { method: 'POST', body: formData });
        if (res.ok) {
            await loadArtifacts(currentSessionId);

            showToast(`📥 Saved to Pending Review - accept it in the dashboard once it's ready.`, "success");
        }
    } catch(e) {
        console.error("Failed to save artifact", e);
    }
}

let currentArtifactFilter = 'all';

function filterArtifacts(type) {
    currentArtifactFilter = type;
    loadArtifacts(currentSessionId);
}

function useArtifact(encodedContent) {
    const content = decodeURIComponent(encodedContent);
    const input = document.getElementById('userInput');
    input.value = `Regarding the artifact "${content}": `;
    input.focus();
}

const ARTIFACT_ICONS = { 'Asset': '🛡️', 'Threat': '⚠️', 'Goal': '🎯', 'Req': '✅' };

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function renderArtifactCard(art) {
    const isApproved = art.status === 'approved';
    const icon = ARTIFACT_ICONS[art.type] || '📌';
    const encodedContent = encodeURIComponent(art.content || '').replace(/'/g, "%27");

    const acceptIcon = isApproved ? '' :
        `<i class="fas fa-check artifact-action-icon accept" onclick="acceptArtifact(${art.id}, event)" title="Accept"></i>`;

    return `
        <div class="artifact-card ${isApproved ? 'approved' : 'pending'} type-${art.type}" onclick="useArtifact('${encodedContent}')">
            <div class="artifact-actions">
                ${acceptIcon}
                <i class="fas fa-pen artifact-action-icon edit" onclick="toggleRefine(${art.id}, event)" title="Refine"></i>
                <i class="fas fa-trash-alt artifact-action-icon discard" onclick="deleteArtifact(${art.id}, event)" title="${isApproved ? 'Remove' : 'Discard'}"></i>
            </div>
            <strong>${icon} ${art.type}</strong><br>
            <div class="artifact-view">
                <span style="color: var(--text-secondary); display: block; margin-top: 4px; padding-right: 60px;">${escapeHtml(art.content)}</span>
            </div>
            <div class="artifact-edit" id="edit-${art.id}" style="display:none; margin-top:6px;" onclick="event.stopPropagation()">
                <textarea rows="3">${escapeHtml(art.content)}</textarea>
                <div style="display:flex; gap:6px; margin-top:6px;">
                    <button class="f-btn" onclick="saveRefinedArtifact(${art.id}, event)">Save</button>
                    <button class="f-btn" onclick="toggleRefine(${art.id}, event)">Cancel</button>
                </div>
            </div>
        </div>
    `;
}

function renderArtifactGroup(title, items) {
    const body = items.length
        ? items.map(renderArtifactCard).join('')
        : `<div class="artifact-group-empty">Nothing here yet.</div>`;
    return `<div class="artifact-group-title">${title}</div>${body}`;
}

// Marks a stepper circle as "done" once the session has at least one approved
// artifact of that type - purely a visual progress indicator, doesn't gate or
// otherwise change navigation (steps stay freely clickable either way).
function updateStepperProgress(artifacts) {
    ['Asset', 'Threat', 'Goal', 'Req'].forEach(type => {
        const hasApproved = artifacts.some(a => a.type === type && a.status === 'approved');
        const stepEl = document.getElementById(`step-${type}`);
        if (stepEl) stepEl.classList.toggle('done', hasApproved);
    });
}

async function loadArtifacts(id) {
    const res = await fetch(`/sessions/${id}/artifacts`);
    const data = await res.json();
    updateStepperProgress(data.artifacts);
    const list = document.getElementById('artifactList');
    list.innerHTML = `
        <div style="margin-bottom:10px; display:flex; gap:5px; flex-wrap:wrap;">
            <button class="mode-btn ${currentArtifactFilter==='all'?'active':''}" onclick="filterArtifacts('all')" style="font-size:0.7em; padding:3px 8px;">All</button>
            <button class="mode-btn ${currentArtifactFilter==='Asset'?'active':''}" onclick="filterArtifacts('Asset')" style="font-size:0.7em; padding:3px 8px;">Assets</button>
            <button class="mode-btn ${currentArtifactFilter==='Threat'?'active':''}" onclick="filterArtifacts('Threat')" style="font-size:0.7em; padding:3px 8px;">Threats</button>
            <button class="mode-btn ${currentArtifactFilter==='Goal'?'active':''}" onclick="filterArtifacts('Goal')" style="font-size:0.7em; padding:3px 8px;">Goals</button>
            <button class="mode-btn ${currentArtifactFilter==='Req'?'active':''}" onclick="filterArtifacts('Req')" style="font-size:0.7em; padding:3px 8px;">Reqs</button>
        </div>
    `;

    const filtered = data.artifacts.filter(a => currentArtifactFilter === 'all' || a.type === currentArtifactFilter);

    if(filtered.length === 0) {
        list.innerHTML += '<div style="color:var(--text-muted); font-size:0.8em; text-align:center;">No artifacts found.</div>';
        return;
    }

    const pending = filtered.filter(a => a.status !== 'approved');
    const approved = filtered.filter(a => a.status === 'approved');

    list.innerHTML += renderArtifactGroup('🕒 Pending Review', pending);
    list.innerHTML += renderArtifactGroup('✅ Approved', approved);
}

async function acceptArtifact(id, event) {
    event.stopPropagation();
    try {
        const res = await fetch(`/artifacts/${id}/approve`, { method: 'POST' });
        if (res.ok) {
            await loadArtifacts(currentSessionId);
            showToast('✅ Artifact accepted and added to the final dashboard!', 'success');
        }
    } catch(e) {
        showToast('Failed to accept artifact.', 'error');
    }
}

function toggleRefine(id, event) {
    event.stopPropagation();
    const editBox = document.getElementById(`edit-${id}`);
    const card = editBox.closest('.artifact-card');
    const viewBox = card.querySelector('.artifact-view');
    const isOpen = editBox.style.display !== 'none';
    editBox.style.display = isOpen ? 'none' : 'block';
    viewBox.style.display = isOpen ? 'block' : 'none';
}

async function saveRefinedArtifact(id, event) {
    event.stopPropagation();
    const editBox = document.getElementById(`edit-${id}`);
    const content = editBox.querySelector('textarea').value.trim();
    if (!content) return;

    const formData = new FormData();
    formData.append('content', content);

    try {
        const res = await fetch(`/artifacts/${id}/edit`, { method: 'POST', body: formData });
        if (res.ok) {
            await loadArtifacts(currentSessionId);
            showToast('✏️ Artifact updated.', 'success');
        }
    } catch(e) {
        showToast('Failed to update artifact.', 'error');
    }
}

async function deleteArtifact(id, event) {
    event.stopPropagation();

    if(!confirm("Do you really want to remove this artifact from the dashboard?")) return;

    try {
        const res = await fetch(`/artifacts/${id}`, { method: 'DELETE' });
        if (res.ok) {
            await loadArtifacts(currentSessionId);
        }
    } catch(e) {
        console.error("Failed to delete artifact:", e);
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// --- DRAG & DROP LOGIC ---
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    let dt = e.dataTransfer;
    let files = dt.files;

    if (files && files.length > 0) {
        fileInput.files = files;
        uploadFile();
    }
});

// --- SIDEBAR COLLAPSE LOGIC ---
function toggleLeftSidebar() {
    const sidebar = document.querySelector('.sidebar-left');
    const icon = document.getElementById('icon-left');

    sidebar.classList.toggle('collapsed');

    if (sidebar.classList.contains('collapsed')) {
        icon.className = 'fas fa-chevron-right';
        showToast("Hide left sidebar", "info");
    } else {
        icon.className = 'fas fa-chevron-left';
    }
}

function toggleRightSidebar() {
    const sidebar = document.querySelector('.sidebar-right');
    const icon = document.getElementById('icon-right');

    sidebar.classList.toggle('collapsed');

    if (sidebar.classList.contains('collapsed')) {
        icon.className = 'fas fa-chevron-left';
        showToast("Hide right sidebar", "info");
    } else {
        icon.className = 'fas fa-chevron-right';
    }
}
