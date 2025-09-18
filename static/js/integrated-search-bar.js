// File: static/js/integrated-search-bar.js
/**
 * IntegratedSearchBar - AGGRESSIVE DUPLICATE PREVENTION VERSION
 * Completely prevents multiple instances and duplicate requests
 */

// GLOBAL SINGLETON PATTERN - Prevents multiple instances
if (window.IntegratedSearchBarInstance) {
    console.log('🚫 IntegratedSearchBar already exists, skipping initialization');
} else {

class IntegratedSearchBar {
    constructor({ formSelector, inputSelector, toggleButtonsSelector, onSubmit }) {
        // Mark as initialized immediately
        window.IntegratedSearchBarInstance = this;
        
        this.form = document.querySelector(formSelector);
        this.input = document.querySelector(inputSelector);
        this.sendBtn = document.querySelector('#send-btn');
        this.toggles = Array.from(document.querySelectorAll(toggleButtonsSelector));
        this.onSubmit = onSubmit;
        this.options = { images: false, drawings: false };
        
        // AGGRESSIVE DUPLICATE PREVENTION
        this.isProcessing = false;
        this.lastQuery = null;
        this.lastRequestTime = 0;
        this.requestInFlight = false;
        
        this.contentRenderer = null;
        this.init();
    }

    init() {
        if (!this.form || !this.input) {
            console.error('❌ Form or input not found');
            return;
        }

        // REMOVE ALL EXISTING EVENT LISTENERS COMPLETELY
        this.cleanupExistingListeners();
        
        this.initializeProfessionalRenderer();

        // Input event listener
        this.input.addEventListener('input', this.handleInput.bind(this));

        // Toggle event listeners
        this.toggles.forEach(btn => {
            btn.addEventListener('click', this.handleToggleClick.bind(this, btn));
        });

        // SINGLE form event listener with aggressive prevention
        this.form.addEventListener('submit', this.handleFormSubmit.bind(this), { 
            capture: true, 
            passive: false 
        });

        console.log('✅ IntegratedSearchBar initialized with aggressive duplicate prevention');
    }

    cleanupExistingListeners() {
        // Clone and replace form to remove ALL existing listeners
        const newForm = this.form.cloneNode(true);
        this.form.parentNode.replaceChild(newForm, this.form);
        this.form = newForm;
        
        // Update references
        this.input = this.form.querySelector('#query-input');
        this.sendBtn = this.form.querySelector('#send-btn');
        
        // Remove any onclick attributes
        this.form.removeAttribute('onsubmit');
        if (this.sendBtn) this.sendBtn.removeAttribute('onclick');
        
        console.log('🧹 Cleaned up existing event listeners');
    }

    handleInput = (e) => {
        if (this.sendBtn) {
            this.sendBtn.disabled = e.target.value.trim() === '';
        }
    }

    handleToggleClick = (button, e) => {
        e.preventDefault();
        const opt = button.dataset.option;
        this.options[opt] = !this.options[opt];
        button.classList.toggle('active', this.options[opt]);
        button.setAttribute('aria-pressed', this.options[opt]);
    }

    handleFormSubmit = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        
        // IMMEDIATE BLOCKING
        if (this.requestInFlight) {
            console.log('🚫 Request blocked: Request already in flight');
            return false;
        }

        this.handleSubmitWithGuards();
        return false;
    }

    handleSubmitWithGuards() {
        const now = Date.now();
        const query = this.input.value.trim();
        
        // GUARD 1: Request in flight check
        if (this.requestInFlight) {
            console.log('🚫 Request blocked: Already in flight');
            return;
        }

        // GUARD 2: Processing check
        if (this.isProcessing) {
            console.log('🚫 Request blocked: Already processing');
            return;
        }

        // GUARD 3: Empty query check
        if (!query) {
            console.log('🚫 Request blocked: Empty query');
            return;
        }

        // GUARD 4: Rapid duplicate check
        if (this.lastQuery === query && (now - this.lastRequestTime) < 1000) {
            console.log('🚫 Request blocked: Duplicate within 1 second');
            return;
        }

        console.log('✅ Request approved:', query);
        this.lastQuery = query;
        this.lastRequestTime = now;
        this.requestInFlight = true;
        
        this.handleSubmit();
    }

    async handleSubmit() {
        this.isProcessing = true;
        
        const query = this.input.value.trim();
        console.log('🚀 SINGLE request starting:', query);

        // Clear existing responses
        const conversation = document.getElementById('conversation');
        if (conversation) {
            const existingThinking = conversation.querySelectorAll('.thinking-indicator');
            existingThinking.forEach(el => el.remove());
        }

        const data = {
            query,
            include_images: this.options.images,
            include_drawings: this.options.drawings
        };

        // Show loading state
        if (this.sendBtn) {
            this.sendBtn.innerHTML = '⏳';
            this.sendBtn.disabled = true;
        }

        // Show thinking indicator
        const thinkingIndicator = this.createThinkingIndicator();
        if (conversation) {
            conversation.appendChild(thinkingIndicator);
            conversation.scrollTop = conversation.scrollHeight;
        }

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            // Remove thinking indicator
            if (thinkingIndicator && thinkingIndicator.parentNode) {
                thinkingIndicator.remove();
            }

            // Render response
            this.renderSingleResponse(result, query);
            
            // Clear input after success
            this.input.value = '';
            console.log('✅ SINGLE request completed successfully');

        } catch (error) {
            console.error('❌ Request failed:', error);
            
            if (thinkingIndicator && thinkingIndicator.parentNode) {
                thinkingIndicator.remove();
            }
            
            this.displayError('Failed to process query. Please try again.');
        } finally {
            // Reset all flags
            if (this.sendBtn) {
                this.sendBtn.innerHTML = '→';
                this.sendBtn.disabled = this.input.value.trim() === '';
            }
            
            setTimeout(() => {
                this.isProcessing = false;
                this.requestInFlight = false;
                console.log('🔓 Ready for next request');
            }, 1000); // 1 second cooldown
        }
    }

    // All your existing methods remain the same
    initializeProfessionalRenderer() {
        if (window.ProfessionalContentRenderer) {
            this.contentRenderer = new window.ProfessionalContentRenderer();
            console.log('✅ Professional Content Renderer loaded');
        } else {
            console.warn('⚠️ Professional Content Renderer not available');
        }
    }

    renderSingleResponse(data, query) {
        const conversation = document.getElementById('conversation');
        const welcomeScreen = document.getElementById('welcome-screen');

        if (welcomeScreen) welcomeScreen.style.display = 'none';
        if (conversation) conversation.style.display = 'block';

        const userMessage = this.createUserMessage(query);
        conversation.appendChild(userMessage);

        if (data.success && data.answer) {
            const assistantMessage = this.createModernAssistantMessage(data);
            conversation.appendChild(assistantMessage);
        } else {
            const errorMessage = this.createErrorMessage(data.error || 'Failed to process query');
            conversation.appendChild(errorMessage);
        }

        conversation.scrollTop = conversation.scrollHeight;
        this.saveToHistory(query);
    }

    createThinkingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message thinking-indicator animate-in';
        indicator.innerHTML = `
            <div class="message-header">
                <div class="assistant-avatar">🤖</div>
                <span class="assistant-name">TEPPL AI</span>
                <span class="message-time">Thinking...</span>
            </div>
            <div class="message-content">
                <div class="thinking-dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
                <p>Searching NCDOT documents...</p>
            </div>
        `;
        return indicator;
    }

    createUserMessage(query) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message animate-in';
        messageElement.innerHTML = `
            <div class="message-header">
                <div class="user-avatar">👤</div>
                <span class="user-name">You</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">
                <p>${this.escapeHtml(query)}</p>
            </div>
        `;
        return messageElement;
    }

    createModernAssistantMessage(data) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message assistant-message animate-in';
        
        const confidence = data.confidence || 0.75;
        const confidenceColor = confidence > 0.8 ? '#28a745' : confidence > 0.6 ? '#ffc107' : '#dc3545';
        const confidenceText = `${Math.round(confidence * 100)}% confidence`;

        messageElement.innerHTML = `
            <div class="message-header">
                <div class="assistant-avatar">🤖</div>
                <span class="assistant-name">TEPPL AI</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
                <span class="confidence-badge" style="background-color: ${confidenceColor}">
                    ${confidenceText}
                </span>
            </div>
            <div class="message-content">
                <div class="assistant-answer">
                    ${this.renderModernMarkdown(data.answer)}
                </div>
                ${data.sources && data.sources.length > 0 ? this.renderSources(data.sources) : ''}
                ${data.images && data.images.length > 0 ? this.renderImages(data.images) : ''}
            </div>
        `;
        return messageElement;
    }

    createErrorMessage(error) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message assistant-message error-message animate-in';
        messageElement.innerHTML = `
            <div class="message-header">
                <div class="assistant-avatar">⚠️</div>
                <span class="assistant-name">TEPPL AI</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">
                <div class="error-content">
                    <p><strong>Error:</strong> ${this.escapeHtml(error)}</p>
                </div>
            </div>
        `;
        return messageElement;
    }

    renderModernMarkdown(text) {
        if (!text) return '';
        let html = text;
        html = html.replace(/^# (.*$)/gm, '<h1 class="markdown-h1">$1</h1>');
        html = html.replace(/^## (.*$)/gm, '<h2 class="markdown-h2">$1</h2>');
        html = html.replace(/^### (.*$)/gm, '<h3 class="markdown-h3">$1</h3>');
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/^• (.*)$/gm, '<li class="markdown-bullet">$1</li>');
        html = html.replace(/(<li class="markdown-bullet">.*<\/li>\s*)+/gs, '<ul class="markdown-list">$&</ul>');
        return this.convertParagraphs(html);
    }

    convertParagraphs(text) {
        const paragraphs = text.split(/\n\s*\n/);
        return paragraphs
            .map(p => p.trim())
            .filter(p => p.length > 0)
            .map(p => {
                if (p.match(/^<(h[1-6]|ul|ol|div|blockquote)/i)) {
                    return p;
                }
                return `<p class="markdown-paragraph">${p.replace(/\n/g, '<br>')}</p>`;
            })
            .join('');
    }

    renderSources(sources) {
        if (!sources || sources.length === 0) return '';
        const sourcesHtml = sources.slice(0, 5).map((source, index) => {
            const title = source.title || 'NCDOT Document';
            const page = source.page || source.pages || 'N/A';
            const relevance = source.relevance || `${Math.round((source.similarity_score || 0.5) * 100)}%`;
            const filename = source.filename || (source.file_path ? source.file_path.split('/').pop() : null);
            const internalLink = source.internal_link || (filename ? `/documents/${filename}` : null);
            
            return `
                <div class="source-item" data-source-id="${index + 1}">
                    <div class="source-header">
                        <span class="source-number">${index + 1}</span>
                        <span class="source-title">${this.escapeHtml(title)}</span>
                        <span class="source-relevance">${relevance}</span>
                    </div>
                    <div class="source-meta">
                        <span class="source-page">Page ${page}</span>
                        ${internalLink ? `<a href="${internalLink}" target="_blank" class="source-link">View Document</a>` : ''}
                    </div>
                    <div class="source-content">
                        ${this.escapeHtml((source.content || '').substring(0, 200))}${source.content && source.content.length > 200 ? '...' : ''}
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="sources-section">
                <h4 class="sources-title">📚 Sources (${sources.length} found)</h4>
                <div class="sources-list">${sourcesHtml}</div>
            </div>
        `;
    }

    renderImages(images) {
        if (!images || images.length === 0) return '';
        const imagesHtml = images.slice(0, 6).map((image, index) => {
            const thumbnailPath = image.thumbnail_path || `thumbnails/${image.id}_thumb.jpg`;
            const altText = image.original_filename || `Technical diagram ${index + 1}`;
            
            return `
                <div class="image-item" data-image-id="${image.id}">
                    <img src="/${thumbnailPath}" alt="${this.escapeHtml(altText)}" class="thumbnail-image" onclick="this.parentElement.classList.toggle('expanded')" onerror="this.style.display='none'">
                    <div class="image-meta">
                        <span class="image-filename">${this.escapeHtml(image.readable_filename || altText)}</span>
                        <span class="image-relevance">${Math.round((image.similarity_score || 0.5) * 100)}%</span>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="images-section">
                <h4 class="images-title">🖼️ Related Images (${images.length} found)</h4>
                <div class="images-grid">${imagesHtml}</div>
            </div>
        `;
    }

    displayError(message) {
        const conversation = document.getElementById('conversation');
        if (conversation) {
            const errorMessage = this.createErrorMessage(message);
            conversation.appendChild(errorMessage);
            conversation.scrollTop = conversation.scrollHeight;
        }
    }

    saveToHistory(query) {
        try {
            let history = JSON.parse(localStorage.getItem('teppl_search_history') || '[]');
            history = history.filter(item => item !== query);
            history.unshift(query);
            history = history.slice(0, 20);
            localStorage.setItem('teppl_search_history', JSON.stringify(history));
        } catch (error) {
            console.warn('Failed to save search history:', error);
        }
    }

    escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Quick prompt functionality
    handleQuickPrompt(promptText) {
        if (this.input) {
            this.input.value = promptText;
            this.input.dispatchEvent(new Event('input'));
            this.input.focus();
        }
    }

    submitQuery(query) {
        if (this.input) {
            this.input.value = query;
            this.handleSubmitWithGuards();
        }
    }

    viewPDF(filePath, page = 1) {
        if (!filePath) return;
        // Quick win: new tab, browser PDF viewer
        window.open(`${filePath}#page=${page}`, '_blank', 'noopener');
    }
}

// SINGLETON INITIALIZATION
document.addEventListener('DOMContentLoaded', function() {
    if (!window.IntegratedSearchBarInstance) {
        window.searchBar = new IntegratedSearchBar({
            formSelector: '#search-form',
            inputSelector: '#query-input',
            toggleButtonsSelector: '[data-option]'
        });
        
        // Setup quick prompt handlers
        const quickPrompts = document.querySelectorAll('.quick-prompt-tile, .prompt-tile');
        quickPrompts.forEach(tile => {
            tile.addEventListener('click', (e) => {
                const promptText = e.target.textContent || e.target.innerText;
                if (window.searchBar && promptText) {
                    window.searchBar.submitQuery(promptText.trim());
                }
            });
        });
        
        console.log('✅ IntegratedSearchBar singleton initialized');
    }
});

// Expose globally
window.IntegratedSearchBar = IntegratedSearchBar;

}
