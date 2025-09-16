// File: static/js/quick-actions-manager.js

/**
 * QuickActionsManager
 * Handles default + custom quick actions in the left pane.
 */

export default class QuickActionsManager {
  constructor({
    listSelector,
    searchInputSelector,
    addButtonSelector,
    storageKey = 'teppl_custom_actions'
  }) {
    this.listEl = document.querySelector(listSelector);
    this.searchInput = document.querySelector(searchInputSelector);
    this.addButton = document.querySelector(addButtonSelector);
    this.storageKey = storageKey;
    this.defaultActions = [
      { id: 'sig-timing', name: 'Signal Timing', query: 'Signal timing standards' },
      { id: 'intx-design', name: 'Intersection Design', query: 'Intersection geometry guidelines' },
      { id: 'sign-placement', name: 'Sign Placement', query: 'Sign placement requirements' }
    ];
    this.customActions = this.loadCustomActions();
    this.filteredActions = [];
    this.init();
  }

  init() {
    this.renderList(this.getAllActions());
    this.searchInput.addEventListener('input', () => this.handleSearch());
    this.addButton.addEventListener('click', () => this.openAddModal());
    this.listEl.addEventListener('click', (e) => this.handleListClick(e));
  }

  loadCustomActions() {
    try {
      const data = JSON.parse(localStorage.getItem(this.storageKey)) || [];
      return Array.isArray(data) ? data : [];
    } catch {
      return [];
    }
  }

  saveCustomActions() {
    localStorage.setItem(this.storageKey, JSON.stringify(this.customActions));
  }

  getAllActions() {
    return [...this.defaultActions, ...this.customActions];
  }

  handleSearch() {
    const q = this.searchInput.value.trim().toLowerCase();
    const matches = this.getAllActions().filter(a => 
      a.name.toLowerCase().includes(q) || a.query.toLowerCase().includes(q)
    );
    this.renderList(matches);
  }

  renderList(actions) {
    this.listEl.innerHTML = '';
    
    if (actions.length === 0) {
      this.listEl.innerHTML = '<div class="no-actions">No actions found</div>';
      return;
    }

    actions.forEach(action => {
      const isCustom = !this.defaultActions.find(def => def.id === action.id);
      const item = document.createElement('div');
      item.className = 'action-item';
      item.dataset.actionId = action.id;
      item.dataset.query = action.query;
      
      item.innerHTML = `
          <div class="action-item-content">
              <span class="action-name">${action.name}</span>
          </div>
          ${isCustom ? `
              <div class="action-item-buttons">
                  <button class="edit-btn" data-action="edit" title="Edit"></button>
                  <button class="delete-btn" data-action="delete" title="Delete"></button>
              </div>
          ` : ''}
      `;
      
      this.listEl.appendChild(item);
    });
  }

  isCustom(id) {
    return this.customActions.some(a => a.id === id);
  }

  handleListClick(e) {
    const actionItem = e.target.closest('.action-item');
    if (!actionItem) return;
    if (e.target.classList.contains('edit-btn')) {
      const actionId = actionItem.dataset.actionId;
      this.editAction(actionId);
      return;
    }
    if (e.target.classList.contains('delete-btn')) {
      const actionId = actionItem.dataset.actionId;
      this.deleteAction(actionId);
      return;
    }
    const query = actionItem.dataset.query;
    if (query && this.onActionSelect) {
      this.onActionSelect(query);
    }
  }

  openAddModal() {
    document.getElementById('add-custom-action').click();
  }

  addCustomAction(actionData) {
    if (!actionData || !actionData.name || !actionData.query) {
      console.error('Invalid action data:', actionData);
      return;
    }
    this.customActions.push({
      id: actionData.id || this._generateId(actionData.name),
      name: actionData.name,
      query: actionData.query
    });
    this.saveCustomActions();
    this.renderList(this.getAllActions());
  }

  openEditModal(actionId) {
    this.editAction(actionId);
  }

  editAction(actionId) {
    const action = this.customActions.find(a => a.id === actionId);
    if (action) {
      // Make sure the modal instance is accessible globally
      const modal = window.quickActionsModal;
      if (modal && typeof modal.showEdit === 'function') {
        modal.showEdit(action);
      }
    }
  }

  _generateId(name) {
    return name.toLowerCase()
      .replace(/[^\w]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }

  deleteAction(id) {
    this.customActions = this.customActions.filter(a => a.id !== id);
    this.saveCustomActions();
    this.handleSearch();
  }

  generateId() {
    return 'custom-' + Date.now();
  }
}
