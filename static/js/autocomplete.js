// static/autocomplete.js
// Enhanced Autocomplete System for TEPPL AI

class AutocompleteSystem {
    constructor() {
        this.queryInput = document.getElementById('query-input');
        this.dropdown = document.getElementById('autocomplete-dropdown');
        if (!this.dropdown) { 
            // No dropdown element included in new layout; disable autocomplete.
            return;
        }
        this.list = document.getElementById('autocomplete-list');
        
        this.searchHistory = JSON.parse(localStorage.getItem('teppl_search_history') || '[]');
        this.customActions = JSON.parse(localStorage.getItem('teppl_custom_actions') || '[]');
        
        this.currentSuggestions = [];
        this.selectedIndex = -1;
        this.isVisible = false;
        
        this.init();
    }
    
    init() {
        if (!this.queryInput) return;
        
        // Input event listeners
        this.queryInput.addEventListener('input', (e) => this.handleInput(e));
        this.queryInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.queryInput.addEventListener('focus', () => this.handleFocus());
        this.queryInput.addEventListener('blur', (e) => this.handleBlur(e));
        
        // Click outside to close
        document.addEventListener('click', (e) => this.handleDocumentClick(e));
    }
    
    handleInput(e) {
        const query = e.target.value.trim();
        
        if (query.length < 2) {
            this.hide();
            return;
        }
        
        const suggestions = this.generateSuggestions(query);
        
        if (suggestions.length > 0) {
            this.show(suggestions);
        } else {
            this.hide();
        }
    }
    
    handleKeydown(e) {
        if (!this.isVisible) return;
        
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.selectNext();
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.selectPrevious();
                break;
            case 'Tab':
                if (this.selectedIndex >= 0) {
                    e.preventDefault();
                    this.applySuggestion(this.currentSuggestions[this.selectedIndex]);
                }
                break;
            case 'Enter':
                if (this.selectedIndex >= 0) {
                    e.preventDefault();
                    this.applySuggestion(this.currentSuggestions[this.selectedIndex]);
                }
                break;
            case 'Escape':
                this.hide();
                break;
        }
    }
    
    handleFocus() {
        const query = this.queryInput.value.trim();
        if (query.length >= 2) {
            const suggestions = this.generateSuggestions(query);
            if (suggestions.length > 0) {
                this.show(suggestions);
            }
        }
    }
    
    handleBlur(e) {
        // Delay hide to allow for click events
        setTimeout(() => {
            if (!this.dropdown.contains(document.activeElement)) {
                this.hide();
            }
        }, 150);
    }
    
    handleDocumentClick(e) {
        if (!this.queryInput.contains(e.target) && !this.dropdown.contains(e.target)) {
            this.hide();
        }
    }
    
    generateSuggestions(query) {
        const suggestions = [];
        const lowerQuery = query.toLowerCase();
        
        // 1. Custom actions (only custom, not default ones)
        const customSuggestions = this.customActions
            .filter(action => 
                action.name.toLowerCase().includes(lowerQuery) ||
                action.query.toLowerCase().includes(lowerQuery)
            )
            .slice(0, 3)
            .map(action => ({
                type: 'custom',
                icon: '⚡',
                title: action.name,
                query: action.query,
                subtitle: 'Custom Action'
            }));
        
        suggestions.push(...customSuggestions);
        
        // 2. Search history (exact matches and partial matches)
        const historySuggestions = this.searchHistory
            .filter(item => item.toLowerCase().includes(lowerQuery))
            .slice(0, 4)
            .map(item => ({
                type: 'history',
                icon: '🕐',
                title: item,
                query: item,
                subtitle: 'Recent Search'
            }));
        
        suggestions.push(...historySuggestions);
        
        // 3. Smart completions based on common patterns
        const smartSuggestions = this.generateSmartSuggestions(query);
        suggestions.push(...smartSuggestions);
        
        // Remove duplicates and limit results
        const uniqueSuggestions = this.removeDuplicates(suggestions);
        return uniqueSuggestions.slice(0, 6);
    }
    
    generateSmartSuggestions(query) {
        const suggestions = [];
        const lowerQuery = query.toLowerCase();
        
        // Common traffic engineering patterns
        const patterns = [
            {
                triggers: ['what are', 'what is'],
                completions: [
                    'What are the MUTCD requirements for',
                    'What are the signal timing standards for',
                    'What are the intersection design guidelines for',
                    'What are the sign placement requirements for'
                ]
            },
            {
                triggers: ['how to', 'how do'],
                completions: [
                    'How to design intersection geometry',
                    'How to calculate signal timing',
                    'How to determine sign placement',
                    'How to analyze traffic safety'
                ]
            },
            {
                triggers: ['show me', 'find'],
                completions: [
                    'Show me technical drawings for',
                    'Show me examples of',
                    'Find specifications for',
                    'Find guidelines for'
                ]
            }
        ];
        
        for (const pattern of patterns) {
            const matchingTrigger = pattern.triggers.find(trigger => 
                lowerQuery.startsWith(trigger)
            );
            
            if (matchingTrigger) {
                const relevantCompletions = pattern.completions
                    .filter(completion => 
                        completion.toLowerCase().startsWith(lowerQuery) ||
                        completion.toLowerCase().includes(lowerQuery)
                    )
                    .slice(0, 2)
                    .map(completion => ({
                        type: 'smart',
                        icon: '💡',
                        title: completion,
                        query: completion,
                        subtitle: 'Suggested Completion'
                    }));
                
                suggestions.push(...relevantCompletions);
            }
        }
        
        return suggestions;
    }
    
    removeDuplicates(suggestions) {
        const seen = new Set();
        return suggestions.filter(suggestion => {
            const key = suggestion.query.toLowerCase();
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
    }
    
    show(suggestions) {
        this.currentSuggestions = suggestions;
        this.selectedIndex = -1;
        this.renderSuggestions();
        this.dropdown.style.display = 'block';
        this.isVisible = true;
    }
    
    hide() {
        this.dropdown.style.display = 'none';
        this.isVisible = false;
        this.selectedIndex = -1;
    }
    
    renderSuggestions() {
        this.list.innerHTML = '';
        
        this.currentSuggestions.forEach((suggestion, index) => {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            item.innerHTML = `
                <div class="autocomplete-icon">${suggestion.icon}</div>
                <div class="autocomplete-content">
                    <div class="autocomplete-title">${this.highlightMatch(suggestion.title)}</div>
                    <div class="autocomplete-subtitle">${suggestion.subtitle}</div>
                </div>
                <div class="autocomplete-key">Tab</div>
            `;
            
            item.addEventListener('mousedown', (e) => {
                e.preventDefault();
                this.applySuggestion(suggestion);
            });
            
            item.addEventListener('mouseenter', () => {
                this.selectedIndex = index;
                this.updateSelection();
            });
            
            this.list.appendChild(item);
        });
    }
    
    highlightMatch(text) {
        const query = this.queryInput.value.trim().toLowerCase();
        if (!query) return text;
        
        const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    selectNext() {
        this.selectedIndex = (this.selectedIndex + 1) % this.currentSuggestions.length;
        this.updateSelection();
    }
    
    selectPrevious() {
        this.selectedIndex = this.selectedIndex <= 0 
            ? this.currentSuggestions.length - 1 
            : this.selectedIndex - 1;
        this.updateSelection();
    }
    
    updateSelection() {
        const items = this.list.querySelectorAll('.autocomplete-item');
        items.forEach((item, index) => {
            item.classList.toggle('selected', index === this.selectedIndex);
        });
    }
    
    applySuggestion(suggestion) {
        this.queryInput.value = suggestion.query;
        this.hide();
        this.queryInput.focus();
        
        // Auto-resize textarea
        this.queryInput.style.height = 'auto';
        this.queryInput.style.height = this.queryInput.scrollHeight + 'px';
        
        // Optional: Auto-submit if it's a complete query
        if (suggestion.type === 'history' || suggestion.type === 'custom') {
            // You can uncomment this to auto-submit
            // setTimeout(() => document.getElementById('query-form').dispatchEvent(new Event('submit')), 100);
        }
    }
    
    // Public methods for external use
    addToHistory(query) {
        if (!query || query.trim().length < 3) return;
        
        const cleanQuery = query.trim();
        
        // Remove duplicates and add to front
        this.searchHistory = this.searchHistory.filter(item => item !== cleanQuery);
        this.searchHistory.unshift(cleanQuery);
        
        // Keep only last 20 searches
        this.searchHistory = this.searchHistory.slice(0, 20);
        
        localStorage.setItem('teppl_search_history', JSON.stringify(this.searchHistory));
    }
    
    updateCustomActions() {
        this.customActions = JSON.parse(localStorage.getItem('teppl_custom_actions') || '[]');
    }
}

// Enhanced CSS for autocomplete
const autocompleteStyles = `
.autocomplete-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    margin-top: var(--space-8);
    max-height: 300px;
    overflow-y: auto;
    z-index: 1000;
}

.autocomplete-list {
    padding: var(--space-8) 0;
}

.autocomplete-item {
    display: flex;
    align-items: center;
    gap: var(--space-12);
    padding: var(--space-12) var(--space-16);
    cursor: pointer;
    transition: background-color var(--duration-fast);
    border-left: 3px solid transparent;
}

.autocomplete-item:hover,
.autocomplete-item.selected {
    background: var(--color-secondary);
    border-left-color: var(--color-primary);
}

.autocomplete-icon {
    font-size: var(--font-size-lg);
    flex-shrink: 0;
    width: 24px;
    text-align: center;
}

.autocomplete-content {
    flex: 1;
    min-width: 0;
}

.autocomplete-title {
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
    color: var(--color-text);
    margin-bottom: var(--space-2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.autocomplete-title mark {
    background: var(--color-primary);
    color: var(--color-btn-primary-text);
    padding: 0 2px;
    border-radius: 2px;
}

.autocomplete-subtitle {
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
}

.autocomplete-key {
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
    background: var(--color-secondary);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
    font-family: var(--font-family-mono);
}

@media (max-width: 768px) {
    .autocomplete-key {
        display: none;
    }
}
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = autocompleteStyles;
document.head.appendChild(styleSheet);

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.autocompleteSystem = new AutocompleteSystem();
    });
} else {
    window.autocompleteSystem = new AutocompleteSystem();
}
