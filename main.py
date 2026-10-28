import time
from typing import List, Optional
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Local AMD Server")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer and lightweight model onto the AMD GPU
print("--- Starting Server ---", flush=True)
print("1. Loading Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
print("2. Downloading/Loading Model Weights...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
).to("cuda")
    
print("3. Success! Model moved to AMD VRAM.", flush=True)
model.eval()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = tokenizer.apply_chat_template(
        messages_dict, tokenize=False, add_generation_prompt=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }]
    }

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Local Chat (Qwen 0.5B)</title>
        <style>
            body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; display: flex; flex-direction: column; height: 100vh; }
            header { background: #1e293b; padding: 14px 20px; font-weight: 600; border-bottom: 1px solid #334155; }
            #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
            .msg { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; }
            .user { align-self: flex-end; background: #2563eb; color: white; }
            .assistant { align-self: flex-start; background: #1e293b; border: 1px solid #334155; }
            #input-form { display: flex; padding: 16px; background: #1e293b; border-top: 1px solid #334155; gap: 10px; }
            input { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: white; font-size: 15px; }
            button { padding: 12px 20px; border-radius: 8px; border: none; background: #3b82f6; color: white; font-weight: 600; cursor: pointer; }
        </style>
    </head>
    <body>
        <header>Qwen2.5-0.5B Local Server (AMD ROCm)</header>
        <div id="chat-box"></div>
        <form id="input-form">
            <input type="text" id="user-input" placeholder="Type a message..." autocomplete="off" required />
            <button type="submit" id="send-btn">Send</button>
        </form>
        <script>
            const chatBox = document.getElementById('chat-box');
            const form = document.getElementById('input-form');
            const input = document.getElementById('user-input');
            const messages = [];

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const text = input.value.trim();
                if (!text) return;

                input.value = '';
                messages.push({ role: 'user', content: text });
                
                const userDiv = document.createElement('div');
                userDiv.className = 'msg user';
                userDiv.textContent = text;
                chatBox.appendChild(userDiv);

                const thinkingDiv = document.createElement('div');
                thinkingDiv.className = 'msg assistant';
                thinkingDiv.textContent = 'Thinking...';
                chatBox.appendChild(thinkingDiv);
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const res = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ messages, max_tokens: 512 })
                    });
                    const data = await res.json();
                    const reply = data.choices[0].message.content;
                    thinkingDiv.textContent = reply;
                    messages.push({ role: 'assistant', content: reply });
                } catch (err) {
                    thinkingDiv.textContent = 'Error communicating with server.';
                }
            });
        </script>
    </body>
    </html>
    """