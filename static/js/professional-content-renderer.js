// Professional Content Renderer for TEPPL AI - COMPLETE VERSION
class ProfessionalContentRenderer {
    constructor() {
        this.markdownPatterns = {
            h1: /^# (.*$)/gm,
            h2: /^## (.*$)/gm,
            h3: /^### (.*$)/gm,
            bold: /\*\*(.*?)\*\*/g,
            italic: /\*(.*?)\*/g,
            bullet: /^• (.*)$/gm,
            code: /`([^`]+)`/g,
            blockquote: /^> (.*)$/gm
        };
        
        this.initializeRenderer();
    }
    
    initializeRenderer() {
        console.log('✅ Professional Content Renderer initialized');
    }
    
    renderMarkdownContent(markdownText) {
        if (!markdownText) return '';
        
        let html = markdownText;
        
        // Convert headers with proper CSS classes
        html = html.replace(this.markdownPatterns.h1, '<h1 class="md-h1">$1</h1>');
        html = html.replace(this.markdownPatterns.h2, '<h2 class="md-h2">$1</h2>');
        html = html.replace(this.markdownPatterns.h3, '<h3 class="md-h3">$1</h3>');
        
        // Convert text formatting
        html = html.replace(this.markdownPatterns.bold, '<strong>$1</strong>');
        html = html.replace(this.markdownPatterns.italic, '<em>$1</em>');
        html = html.replace(this.markdownPatterns.code, '<code>$1</code>');
        
        // Convert bullet points with proper styling
        html = html.replace(this.markdownPatterns.bullet, '<div class="md-bullet">$1</div>');
        
        // Convert blockquotes
        html = html.replace(this.markdownPatterns.blockquote, '<blockquote class="md-blockquote">$1</blockquote>');
        
        // Convert line breaks to paragraphs
        html = this.convertParagraphs(html);
        
        return `<div class="markdown-content professional-content">${html}</div>`;
    }
    
    convertParagraphs(html) {
        // Split by double line breaks and wrap in paragraph tags
        const paragraphs = html.split('\n\n').filter(p => p.trim());
        return paragraphs.map(p => {
            // Skip if already wrapped in HTML tags
            if (p.startsWith('<')) return p;
            return `<div class="md-paragraph">${p.trim()}</div>`;
        }).join('');
    }
    
    createProfessionalMessage(data) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message assistant-message animate-in';
        
        const avatarElement = document.createElement('div');
        avatarElement.className = 'message-avatar';
        avatarElement.innerHTML = '🤖';
        
        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        
        // Render markdown content
        const markdownHtml = this.renderMarkdownContent(data.answer);
        contentElement.innerHTML = markdownHtml;
        
        // Add confidence indicator
        if (data.confidence) {
            const confidenceElement = this.createConfidenceIndicator(data.confidence);
            contentElement.appendChild(confidenceElement);
        }
        
        // Add collapsible sources
        if (data.sources && data.sources.length > 0) {
            const sourcesElement = this.createCollapsibleSources(data.sources);
            contentElement.appendChild(sourcesElement);
        }
        
        messageElement.appendChild(avatarElement);
        messageElement.appendChild(contentElement);
        
        return messageElement;
    }
    
    createConfidenceIndicator(confidence) {
        const confidenceElement = document.createElement('div');
        confidenceElement.className = 'confidence-indicator';
        
        const percentage = Math.round(confidence * 100);
        const icon = percentage >= 80 ? '🎯' : percentage >= 70 ? '📊' : '⚡';
        
        confidenceElement.innerHTML = `
            ${icon} <strong>${percentage}% Confidence</strong>
        `;
        
        return confidenceElement;
    }
    
    createCollapsibleSources(sources) {
        const sourcesContainer = document.createElement('div');
        sourcesContainer.className = 'sources-section-collapsible';
        
        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.className = 'sources-toggle';
        toggleButton.innerHTML = `
            <span class="sources-count">${sources.length} source${sources.length > 1 ? 's' : ''}</span>
            <i class="toggle-icon">▼</i>
        `;
        
        // Create collapsible content
        const sourcesContent = document.createElement('div');
        sourcesContent.className = 'sources-content';
        sourcesContent.style.maxHeight = '0';
        sourcesContent.style.overflow = 'hidden';
        sourcesContent.style.transition = 'max-height 0.3s ease';
        
        // Populate sources
        sources.forEach((source, index) => {
            const sourceElement = this.createSourceCard(source, index + 1);
            sourcesContent.appendChild(sourceElement);
        });
        
        // Toggle functionality
        let isExpanded = false;
        toggleButton.addEventListener('click', () => {
            isExpanded = !isExpanded;
            
            if (isExpanded) {
                sourcesContent.style.maxHeight = sourcesContent.scrollHeight + 'px';
                toggleButton.querySelector('.toggle-icon').textContent = '▲';
                toggleButton.classList.add('expanded');
            } else {
                sourcesContent.style.maxHeight = '0';
                toggleButton.querySelector('.toggle-icon').textContent = '▼';
                toggleButton.classList.remove('expanded');
            }
        });
        
        sourcesContainer.appendChild(toggleButton);
        sourcesContainer.appendChild(sourcesContent);
        
        return sourcesContainer;
    }
    
    createSourceCard(source, index) {
        const sourceCard = document.createElement('div');
        sourceCard.className = 'source-card-enhanced';
        
        // Handle both old and new source formats
        const title = source.title || source.document_title || 'TEPPL Document';
        const pageInfo = source.page_info ? source.page_info.display : `Page ${source.pages || 'N/A'}`;
        const docType = source.document_type || 'Policy Document';
        const relevance = source.relevance || '0%';
        
        sourceCard.innerHTML = `
            <div class="source-header">
                <div class="source-number">${index}</div>
                <div class="source-info">
                    <div class="source-title-enhanced">${title}</div>
                    <div class="source-meta">
                        ${pageInfo} • ${docType} • ${relevance} relevant
                    </div>
                </div>
                <div class="source-actions">
                    ${source.internal_link ? `
                        <a href="${source.internal_link}" class="source-link internal" title="Open Document">
                            📄
                        </a>
                    ` : ''}
                    ${source.external_link ? `
                        <a href="${source.external_link}" class="source-link external" target="_blank" title="External Link">
                            🔗
                        </a>
                    ` : ''}
                </div>
            </div>
            <div class="source-content-preview">
                ${source.content.substring(0, 200)}${source.content.length > 200 ? '...' : ''}
            </div>
        `;
        
        return sourceCard;
    }
}

// Make available globally (NO export syntax)
window.ProfessionalContentRenderer = ProfessionalContentRenderer;