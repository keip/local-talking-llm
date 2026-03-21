import asyncio
import json
import queue
import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
event_queue: queue.Queue = queue.Queue()
_connected: set[WebSocket] = set()
_on_message = None


def set_message_handler(handler) -> None:
    """Register a callback for text messages from the web UI."""
    global _on_message
    _on_message = handler

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Local Talking LLM</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 16px 24px;
    background: #1a1d27;
    border-bottom: 1px solid #2d3148;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  header h1 { font-size: 18px; font-weight: 600; color: #a5b4fc; }
  .meta { font-size: 12px; color: #64748b; display: flex; gap: 12px; flex-wrap: wrap; }
  .meta span { background: #1e2235; padding: 2px 8px; border-radius: 4px; }
  .status-bar {
    padding: 10px 24px;
    background: #13151f;
    border-bottom: 1px solid #1e2235;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #4b5563;
    flex-shrink: 0;
    transition: background 0.3s;
  }
  .status-dot.listening  { background: #6b7280; }
  .status-dot.recording  { background: #ef4444; animation: pulse 1s infinite; }
  .status-dot.processing { background: #f59e0b; animation: pulse 1s infinite; }
  .status-dot.speaking   { background: #10b981; animation: pulse 1s infinite; }
  .status-dot.wake_detected { background: #3b82f6; animation: pulse 0.5s infinite; }
  .status-dot.disconnected  { background: #374151; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .status-label {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8;
  }
  .conversation {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .message {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 15px;
    line-height: 1.5;
    word-break: break-word;
    animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }
  .message.user {
    background: #1e3a5f;
    color: #bfdbfe;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }
  .message.assistant {
    background: #1a1d27;
    color: #e2e8f0;
    border: 1px solid #2d3148;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
  }
  .message .role {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 4px;
  }
  .empty-state {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #374151;
    font-size: 14px;
  }
  .thinking-dots {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 2px;
  }
  .thinking-dots .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #818cf8;
    animation: bounce 0.6s ease-in-out infinite;
  }
  .thinking-dots .dot:nth-child(2) { animation-delay: 0.15s; }
  .thinking-dots .dot:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce {
    0%, 100% { transform: translateY(0); opacity: 0.5; }
    50% { transform: translateY(-6px); opacity: 1; }
  }
  .tool-indicator {
    max-width: 75%;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 13px;
    background: #1c1f2e;
    border: 1px solid #3b3f5c;
    color: #a78bfa;
    align-self: flex-start;
    animation: fadeIn 0.2s ease;
  }
  .tool-indicator .tool-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 2px;
  }
  .tool-history {
    border-top: 1px solid #1e2235;
    background: #0d0f15;
    max-height: 200px;
    overflow-y: auto;
    padding: 10px 24px;
    font-size: 12px;
    color: #64748b;
  }
  .tool-history summary {
    cursor: pointer;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #94a3b8;
    font-size: 11px;
    margin-bottom: 6px;
  }
  .tool-history .entry {
    padding: 4px 0;
    border-bottom: 1px solid #1a1d27;
  }
  .tool-history .entry .name { color: #a78bfa; font-weight: 500; }
  .tool-history .entry .summary { color: #64748b; }
  .input-bar {
    padding: 12px 24px;
    background: #1a1d27;
    border-top: 1px solid #2d3148;
    display: flex;
    gap: 8px;
  }
  .input-bar input {
    flex: 1;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #2d3148;
    background: #0f1117;
    color: #e2e8f0;
    font-size: 14px;
    outline: none;
  }
  .input-bar input:focus { border-color: #a5b4fc; }
  .input-bar input:disabled { opacity: 0.5; }
  .input-bar button {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    background: #4f46e5;
    color: #fff;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
  }
  .input-bar button:hover { background: #4338ca; }
  .input-bar button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<header>
  <h1>Local Talking LLM</h1>
  <div class="meta">
    <span id="meta-model">model: —</span>
    <span id="meta-mode">mode: —</span>
    <span id="meta-tts">tts: —</span>
  </div>
</header>
<div class="status-bar">
  <div class="status-dot disconnected" id="status-dot"></div>
  <span class="status-label" id="status-label">Connecting...</span>
</div>
<div class="conversation" id="conversation">
  <div class="empty-state" id="empty-state">Waiting for conversation...</div>
</div>
<div class="input-bar">
  <input type="text" id="text-input" placeholder="Type a message..." autocomplete="off">
  <button id="send-btn">Send</button>
</div>
<div class="tool-history">
  <details>
    <summary>Tool History</summary>
    <div id="tool-history-list"></div>
  </details>
</div>
<script>
  const dot = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  const conv = document.getElementById('conversation');
  const empty = document.getElementById('empty-state');
  const toolHistoryList = document.getElementById('tool-history-list');
  const textInput = document.getElementById('text-input');
  const sendBtn = document.getElementById('send-btn');
  let currentWs = null;

  const STATE_LABELS = {
    listening: 'Listening for wake phrase',
    recording: 'Recording',
    processing: 'Processing',
    speaking: 'Speaking',
    wake_detected: 'Wake phrase detected',
  };

  let thinkingEl = null;
  let toolIndicatorEl = null;

  function setStatus(state) {
    dot.className = 'status-dot ' + state;
    label.textContent = STATE_LABELS[state] || state;
  }

  function showThinking(afterEl) {
    removeThinking();
    const div = document.createElement('div');
    div.className = 'message assistant';
    const roleLabel = document.createElement('div');
    roleLabel.className = 'role';
    roleLabel.textContent = 'Morgan';
    div.appendChild(roleLabel);
    const dots = document.createElement('div');
    dots.className = 'thinking-dots';
    dots.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    div.appendChild(dots);
    afterEl.after(div);
    conv.scrollTop = conv.scrollHeight;
    thinkingEl = div;
  }

  function removeThinking() {
    if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
  }

  function addMessage(role, text) {
    empty.style.display = 'none';
    const div = document.createElement('div');
    div.className = 'message ' + role;
    const roleLabel = document.createElement('div');
    roleLabel.className = 'role';
    roleLabel.textContent = role === 'user' ? 'You' : 'Morgan';
    div.appendChild(roleLabel);
    const body = document.createElement('div');
    body.textContent = text;
    div.appendChild(body);
    conv.appendChild(div);
    conv.scrollTop = conv.scrollHeight;
    return div;
  }

  function showToolIndicator(toolName) {
    removeToolIndicator();
    empty.style.display = 'none';
    const div = document.createElement('div');
    div.className = 'tool-indicator';
    const lbl = document.createElement('div');
    lbl.className = 'tool-label';
    lbl.textContent = 'Tool';
    div.appendChild(lbl);
    const body = document.createElement('div');
    body.textContent = 'Using ' + toolName + '...';
    div.appendChild(body);
    conv.appendChild(div);
    conv.scrollTop = conv.scrollHeight;
    toolIndicatorEl = div;
  }

  function removeToolIndicator() {
    if (toolIndicatorEl) { toolIndicatorEl.remove(); toolIndicatorEl = null; }
  }

  function addToolHistoryEntry(toolName, summary) {
    const div = document.createElement('div');
    div.className = 'entry';
    div.innerHTML = '<span class="name">' + toolName + '</span>: <span class="summary">' +
      (summary || '(no result)').substring(0, 120) + '</span>';
    toolHistoryList.prepend(div);
  }

  function sendMessage() {
    const text = textInput.value.trim();
    if (!text || !currentWs || currentWs.readyState !== WebSocket.OPEN) return;
    currentWs.send(JSON.stringify({type: 'message', text: text}));
    textInput.value = '';
    textInput.disabled = true;
    sendBtn.disabled = true;
  }

  sendBtn.addEventListener('click', sendMessage);
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  function connect() {
    const ws = new WebSocket('ws://' + location.host + '/ws');
    currentWs = ws;

    ws.onopen = () => setStatus('listening');

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'info') {
        document.getElementById('meta-model').textContent = 'model: ' + msg.model;
        document.getElementById('meta-mode').textContent = 'mode: ' + msg.mode;
        document.getElementById('meta-tts').textContent = 'tts: ' + msg.tts_url;
      } else if (msg.type === 'state') {
        setStatus(msg.state);
      } else if (msg.type === 'tool_call') {
        removeThinking();
        showToolIndicator(msg.tool);
      } else if (msg.type === 'tool_result') {
        removeToolIndicator();
        addToolHistoryEntry(msg.tool, msg.summary);
      } else if (msg.type === 'message') {
        removeToolIndicator();
        if (msg.role === 'assistant') {
          removeThinking();
          textInput.disabled = false;
          sendBtn.disabled = false;
          textInput.focus();
        }
        const el = addMessage(msg.role, msg.text);
        if (msg.role === 'user') showThinking(el);
      }
    };

    ws.onclose = () => {
      dot.className = 'status-dot disconnected';
      label.textContent = 'Disconnected — reconnecting...';
      setTimeout(connect, 1000);
    };

    ws.onerror = () => ws.close();
  }

  connect();
</script>
</body>
</html>
"""


def emit(event: dict) -> None:
    """Thread-safe: put an event into the broadcast queue."""
    event_queue.put_nowait(event)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _connected.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "message" and _on_message:
                text = msg.get("text", "").strip()
                if text:
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, _on_message, text)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _connected.discard(ws)


async def _broadcaster():
    """Background task: drain event_queue and broadcast to all clients."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            event = await loop.run_in_executor(None, lambda: event_queue.get(timeout=0.05))
            payload = json.dumps(event)
            dead = set()
            for ws in list(_connected):
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            _connected.difference_update(dead)
        except queue.Empty:
            pass
        except Exception:
            await asyncio.sleep(0.05)


@app.on_event("startup")
async def _start_broadcaster():
    asyncio.create_task(_broadcaster())


def start_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the web UI server in a background daemon thread."""
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
