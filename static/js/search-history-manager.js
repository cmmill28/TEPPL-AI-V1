// File: static/js/search-history-manager.js

/**
 * SearchHistoryManager
 * Manages searchable history functionality in the left pane.
 */

export default class SearchHistoryManager {
  constructor({ inputSelector, resultsContainerSelector, historyKey = 'teppl_search_history', debounceTime = 300 }) {
    this.input = document.querySelector(inputSelector);
    this.resultsContainer = document.querySelector(resultsContainerSelector);
    this.storageKey = historyKey;
    this.debounceTime = debounceTime;
    this.history = this.loadHistory();
    this.debounceTimer = null;
    this.onSelect = null; // callback when a history item is clicked
    this.init();
  }

  init() {
    if (!this.input || !this.resultsContainer) return;
    this.input.addEventListener('input', (e) => this.handleInput(e));
    this.resultsContainer.addEventListener('click', (e) => this.handleClick(e));
  }

  loadHistory() {
    try {
      const data = JSON.parse(localStorage.getItem(this.storageKey)) || [];
      return Array.isArray(data) ? data : [];
    } catch {
      return [];
    }
  }

  saveHistory(history) {
    localStorage.setItem(this.storageKey, JSON.stringify(history));
    this.history = history;
  }

  handleInput(e) {
    const query = e.target.value.trim().toLowerCase();
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.filterAndRender(query);
    }, this.debounceTime);
  }

  filterAndRender(query) {
    const results = query.length === 0
      ? this.history.slice().reverse()
      : this.fuzzyFilter(this.history, query);
    this.renderResults(results);
  }

  fuzzyFilter(list, query) {
    // Simple fuzzy matching: include if all chars in order
    return list.filter(item => {
      const s = item.toLowerCase();
      let qi = 0;
      for (let i = 0; i < s.length && qi < query.length; i++) {
        if (s[i] === query[qi]) qi++;
      }
      return qi === query.length;
    }).reverse();
  }

  renderResults(results) {
    this.resultsContainer.innerHTML = '';
    if (results.length === 0) {
      this.resultsContainer.innerHTML = '<div class="no-history">No matching history</div>';
      return;
    }
    results.forEach(item => {
      const div = document.createElement('div');
      div.className = 'history-item searchable';
      div.textContent = item;
      this.resultsContainer.appendChild(div);
    });
  }

  handleClick(e) {
    const itemEl = e.target.closest('.history-item');
    if (!itemEl) return;
    const value = itemEl.textContent;
    if (typeof this.onSelect === 'function') {
      this.onSelect(value);
    }
  }
}
