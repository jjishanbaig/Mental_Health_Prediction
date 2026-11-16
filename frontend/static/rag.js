
const chatBox = document.getElementById("chat-box");
const isEmbedded = window.self !== window.top || new URLSearchParams(window.location.search).get('embedded') === '1';
if (isEmbedded) {
    document.getElementById('back-dashboard').hidden = true;
    document.body.classList.add('embedded');
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') window.parent.postMessage({type:'close-knowledge-assistant'}, window.location.origin);
    });
}

function addMessage(message, type) {
    document.getElementById('chat-welcome').hidden = true;
    const div = document.createElement("div");

    div.className = `message ${type}`;

    // textContent prevents document text from being
    // interpreted as executable HTML.
    div.textContent = message;

    chatBox.appendChild(div);
    div.scrollIntoView({block:'nearest'});

    return div;
}


async function uploadDocument() {
    const fileInput = document.getElementById("document");
    const status = document.getElementById("upload-status");

    const file = fileInput.files[0];

    if (!file) {
        status.textContent = "Please select a document.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    status.textContent = "Processing document...";

    try {
        const response = await fetch("/api/rag/upload", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Upload failed");
        }

        status.textContent =
            `${result.filename}: ${result.chunks_created} chunks indexed.`;

        fileInput.value = "";

        loadDocuments();

    } catch (error) {
        status.textContent = error.message;
    }
}


async function loadDocuments() {
    const list = document.getElementById("documents");

    try {
        const response = await fetch("/api/rag/documents");
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Could not load documents");
        }

        list.replaceChildren();

        result.documents.forEach(doc => {
            const li = document.createElement("li");

            const name = document.createElement("span");
            name.textContent =
                `${doc.filename} (${doc.chunks} chunks) `;

            const button = document.createElement("button");
            button.textContent = "Delete";

            button.addEventListener("click", () => {
                removeDocument(doc.document_id);
            });

            if (doc.bundled) {
                name.textContent += '(shared guide)';
                li.append(name);
            } else {
                li.append(name, button);
            }
            list.appendChild(li);
        });

        if (result.documents.length === 0) {
            list.textContent = "No documents uploaded yet.";
        }

    } catch (error) {
        list.textContent = error.message;
    }
}


async function removeDocument(documentId) {
    if (!confirm("Delete this document from the knowledge base?")) {
        return;
    }

    try {
        const response = await fetch(
            `/api/rag/documents/${encodeURIComponent(documentId)}`,
            { method: "DELETE" }
        );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Delete failed");
        }

        loadDocuments();

    } catch (error) {
        alert(error.message);
    }
}


async function askQuestion() {
    const input = document.getElementById("question");
    const status = document.getElementById("chat-status");

    const question = input.value.trim();

    if (!question) {
        status.textContent = "Please enter a question.";
        return;
    }

    input.value = "";

    addMessage("You: " + question, "user");

    const loading = addMessage(
        "AI is searching the knowledge base...",
        "assistant"
    );

    status.textContent = "";

    try {
        const response = await fetch("/api/rag/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Question failed");
        }

        loading.textContent = "AI: " + result.answer;

        if (result.sources && result.sources.length) {
            const sourceText = result.sources.map(source => {
                const page = source.page
                    ? `, page ${source.page}`
                    : "";

                const entry = source.entry_id ? `, ${source.entry_id}` : '';
                return `[Source ${source.source_id}] ${source.filename}${page}${entry}`;
            }).join("\n");

            const sourceDiv = document.createElement("div");
            sourceDiv.className = "source";
            sourceDiv.textContent = "Sources:\n" + sourceText;

            loading.appendChild(sourceDiv);
        }

    } catch (error) {
        loading.textContent = "Error: " + error.message;
    }
}


document.getElementById("question").addEventListener(
    "keydown",
    event => {
        if (event.key === "Enter") {
            askQuestion();
        }
    }
);

document.querySelectorAll('[data-question]').forEach(button => {
    button.addEventListener('click', () => {
        const input = document.getElementById('question');
        input.value = button.dataset.question;
        input.focus();
    });
});


async function initializeAssistant() {
    const status = document.getElementById("rag-status");
    try {
        const response = await fetch("/api/rag/status");
        if (!response.ok) throw new Error("RAG backend is unavailable. Restart the backend after installing its requirements.");
        const result = await response.json();
        status.textContent = result.message;
        status.hidden = isEmbedded && (result.configured || result.guide_available);
    } catch (error) {
        status.textContent = error.message;
    }
    if (!isEmbedded) loadDocuments();
}
initializeAssistant();
