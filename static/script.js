// Constants
const CHAT_CONFIG = {
    API_URL: 'http://localhost:8000',
    BOT_NAME: 'MyAyur Health Assistant',
    TYPING_DELAY: 300,
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
};

class ChatInterface {
    constructor() {
        this.chatContainer = null;
        this.chatBox = null;
        this.userInput = null;
        this.sendButton = null;
        this.loadingIndicator = null;
        this.chatbotIcon = null;
        this.minimizeButton = null;
        this.messageQueue = [];
        this.isProcessing = false;
        this.isMinimized = true;
    }

    initialize() {
    document.addEventListener('DOMContentLoaded', () => {
        this.setupDOMElements();
        this.createLoadingIndicator();
        this.setupEventListeners();
        this.setupErrorHandling();
        this.adjustTextareaHeight();

        // Automatically open the chat on page load
        this.toggleChat();  // This will trigger the chat to pop up when the page loads
    });
}

    setupDOMElements() {
        this.chatContainer = document.getElementById("chat-container");
        this.chatBox = document.getElementById("chat-box");
        this.userInput = document.getElementById("user-input");
        this.sendButton = document.getElementById("send-button");
        this.chatbotIcon = document.getElementById("chatbot-icon");
        this.minimizeButton = document.querySelector(".minimize-button");
    }

    createLoadingIndicator() {
        this.loadingIndicator = document.createElement("div");
        this.loadingIndicator.className = "message bot-message loading-message fade-in";
        this.loadingIndicator.innerHTML = `
            <div class="loading-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span class="loading-text">${CHAT_CONFIG.BOT_NAME} is thinking...</span>
            </div>
        `;
    }

    setupEventListeners() {
        // Chat toggle
        this.chatbotIcon.addEventListener("click", () => this.toggleChat());
        this.minimizeButton.addEventListener("click", () => this.toggleChat());

        // Input handling
        this.userInput.addEventListener("keypress", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                this.handleUserInput();
            }
        });

        // Input resize
        this.userInput.addEventListener("input", () => this.adjustTextareaHeight());

        // Send button
        this.sendButton.addEventListener("click", () => this.handleUserInput());

        // Scroll handling
        this.chatBox.addEventListener("scroll", this.handleScroll.bind(this));
    }

    adjustTextareaHeight() {
        this.userInput.style.height = 'auto';
        this.userInput.style.height = Math.min(this.userInput.scrollHeight, 120) + 'px';
    }

    toggleChat() {
        this.isMinimized = !this.isMinimized;
        
        if (this.isMinimized) {
            this.chatContainer.classList.add('fade-out');
            setTimeout(() => {
                this.chatContainer.style.display = 'none';
                this.chatbotIcon.style.display = 'flex';
            }, 300);
        } else {
            this.chatContainer.style.display = 'flex';
            this.chatbotIcon.style.display = 'none';
            // Force reflow
            this.chatContainer.offsetHeight;
            this.chatContainer.classList.remove('fade-out');
            
            if (!this.hasInitialMessage) {
                this.appendMessage({
                    text: `Welcome! How may I assist you today?`,
                    sender: "bot",
                    type: "greeting"
                });
                this.hasInitialMessage = true;
            }
            
            this.userInput.focus();
        }
    }

    async handleUserInput() {
        const message = this.userInput.value.trim();
        if (!message) return;

        try {
            this.setInputState(false);
            
            this.appendMessage({
                text: message,
                sender: "user"
            });

            this.userInput.value = "";
            this.adjustTextareaHeight();

            this.showLoadingIndicator();
            await this.processMessage(message);

        } catch (error) {
            this.handleError(error);
        } finally {
            this.setInputState(true);
            this.hideLoadingIndicator();
        }
    }

    async processMessage(message) {
        let retries = 0;
        while (retries < CHAT_CONFIG.MAX_RETRIES) {
            try {
                const response = await this.sendMessageToServer(message);
                await this.handleServerResponse(response);
                break;
            } catch (error) {
                retries++;
                if (retries === CHAT_CONFIG.MAX_RETRIES) {
                    throw error;
                }
                await new Promise(resolve => setTimeout(resolve, CHAT_CONFIG.RETRY_DELAY));
            }
        }
    }

    async sendMessageToServer(message) {
        const response = await fetch(`${CHAT_CONFIG.API_URL}/ask/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: message })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        return response;
    }

    async handleServerResponse(response) {
        const botMessageDiv = this.appendMessage({
            text: "",
            sender: "bot",
            isStreaming: true
        });

        const botContent = botMessageDiv.querySelector("span") || botMessageDiv;
        let responseText = "";

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.error) throw new Error(data.error);
                    if (data.token) {
                        responseText += data.token;
                        botContent.textContent = responseText;
                        this.scrollToBottom();
                    }
                }
            }
        }
    }

    appendMessage({ text, sender, isStreaming = false, type = "message" }) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", `${sender}-message`, `message-${type}`, "fade-in");
        
        if (sender === "bot" && isStreaming) {
            const contentSpan = document.createElement("span");
            messageDiv.appendChild(document.createTextNode(`${CHAT_CONFIG.BOT_NAME}: `));
            messageDiv.appendChild(contentSpan);
        } else {
            messageDiv.textContent = `${sender === "user" ? "You" : CHAT_CONFIG.BOT_NAME}: ${text}`;
        }
        
        this.chatBox.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }

    showLoadingIndicator() {
        this.hideLoadingIndicator();
        this.loadingIndicator.classList.remove("fade-out");
        this.chatBox.appendChild(this.loadingIndicator);
        this.scrollToBottom();
    }

    hideLoadingIndicator() {
        if (this.loadingIndicator.parentNode === this.chatBox) {
            this.loadingIndicator.classList.add("fade-out");
            setTimeout(() => {
                if (this.loadingIndicator.parentNode === this.chatBox) {
                    this.chatBox.removeChild(this.loadingIndicator);
                }
            }, 300);
        }
    }

    setInputState(enabled) {
        this.userInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
        if (enabled) this.userInput.focus();
    }

    scrollToBottom() {
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
    }

    handleScroll() {
        const isNearTop = this.chatBox.scrollTop < 50;
        if (isNearTop && !this.isLoadingHistory) {
            // Load more history if needed
        }
    }

    handleError(error) {
        console.error('Chat Error:', error);
        this.appendMessage({
            text: "I apologize, but I encountered an error. Please try again.",
            sender: "bot",
            type: "error"
        });
    }
}

// Initialize chat interface
const chatInterface = new ChatInterface();
chatInterface.initialize();

// Export for external use
window.chatInterface = chatInterface;
