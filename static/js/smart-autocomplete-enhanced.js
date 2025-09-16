// File: static/js/smart-autocomplete-enhanced.js

/**
 * SmartAutocompleteEnhanced
 * Standalone autocomplete system with option-aware suggestions
 */
export default class SmartAutocompleteEnhanced {
  constructor(opts = {}) {
    this.queryInput = document.getElementById('query-input');
    this.dropdown = document.getElementById('autocomplete-dropdown');
    this.list = document.getElementById('autocomplete-list');
    this.includeImages = false;
    this.includeDrawings = false;
    this.currentSuggestions = [];
    this.selectedIndex = -1;
    this.isVisible = false;
    this.init();
  }

  init() {
    if (!this.queryInput || !this.dropdown || !this.list) return;
    
    this.queryInput.addEventListener('input', (e) => this.handleInput(e));
    this.queryInput.addEventListener('keydown', (e) => this.handleKeydown(e));
    this.queryInput.addEventListener('focus', () => this.handleFocus());
    this.queryInput.addEventListener('blur', (e) => this.handleBlur(e));
    
    // Click outside to close
    document.addEventListener('click', (e) => this.handleDocumentClick(e));
    console.log('✅ Enhanced autocomplete initialized');
  }

  setOptions({ include_images, include_drawings }) {
    this.includeImages = include_images;
    this.includeDrawings = include_drawings;
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
    const lowerQuery = query.toLowerCase();
    const suggestions = [];
    
    // Basic smart suggestions
    const patterns = [
      'MUTCD sign requirements for ' + query,
      'Signal timing standards for ' + query,
      'Intersection design guidelines for ' + query
    ];
    
    patterns.forEach(pattern => {
      suggestions.push({
        type: 'smart',
        icon: '💡',
        title: pattern,
        query: pattern,
        subtitle: 'Suggested'
      });
    });
    
    // Boost image/drawing related suggestions if options are enabled
    if (this.includeImages || this.includeDrawings) {
      suggestions.unshift({
        type: 'smart',
        icon: '🖼️',
        title: 'Show me technical drawings for ' + query,
        query: 'Show me technical drawings for ' + query,
        subtitle: 'Visual Content',
        priority: 1
      });
    }
    
    return suggestions.slice(0, 5);
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
        <div class="autocomplete-item-icon">${suggestion.icon}</div>
        <div class="autocomplete-item-content">
          <div class="autocomplete-item-title">${suggestion.title}</div>
          <div class="autocomplete-item-subtitle">${suggestion.subtitle}</div>
        </div>
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
  }

  attachOptionListener(searchBarInstance) {
    // listen for toggles from IntegratedSearchBar
    searchBarInstance.onSubmit = ({ include_images, include_drawings }) => {
      this.setOptions({ include_images, include_drawings });
    };
  }
}
