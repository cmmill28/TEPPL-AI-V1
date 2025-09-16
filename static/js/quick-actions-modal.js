// File: static/quick-actions-modal.js

export default class QuickActionsModal {
  constructor({
    modalSelector,
    openButtonSelector,
    closeButtonSelector,
    formSelector,
    nameInputSelector,
    queryInputSelector,
    errorContainerSelector,
    onSubmit
  }) {
    this.modal = document.querySelector(modalSelector);
    this.openButton = document.querySelector(openButtonSelector);
    this.closeButton = document.querySelector(closeButtonSelector);
    this.form = document.querySelector(formSelector);
    this.nameInput = document.querySelector(nameInputSelector);
    this.queryInput = document.querySelector(queryInputSelector);
    this.errorContainer = document.querySelector(errorContainerSelector);
    this.onSubmit = onSubmit;

    // Track edit mode status
    this.editMode = false;
    this.editingId = null;

    this._bindEvents();
  }

  _bindEvents() {
    if (this.openButton) {
      this.openButton.addEventListener('click', () => this.show());
    }
    if (this.closeButton) {
      this.closeButton.addEventListener('click', () => this.hide());
    }
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && this.modal.classList.contains('active')) {
        this.hide();
      }
    });
    if (this.form) {
      this.form.addEventListener('submit', e => {
        e.preventDefault();
        this._handleFormSubmit();
      });
    }
    // Click outside to close
    this.modal.addEventListener('click', e => {
      if (e.target === this.modal) {
        this.hide();
      }
    });
  }

  showEdit(actionData) {
    this.editMode = true;
    this.editingId = actionData.id;
    this.errorContainer.textContent = '';
    this.nameInput.value = actionData.name;
    this.queryInput.value = actionData.query;
    this.modal.classList.add('active');
    this.nameInput.focus();
  }

  show() {
    this.editMode = false;
    this.editingId = null;
    this.errorContainer.textContent = '';
    this.nameInput.value = '';
    this.queryInput.value = '';
    this.modal.classList.add('active');
    this.nameInput.focus();
  }

  hide() {
    this.modal.classList.remove('active');
  }

  _handleFormSubmit() {
    const name = this.nameInput.value.trim();
    const query = this.queryInput.value.trim();
    if (!name || !query) {
      this.errorContainer.textContent = 'Please enter both a name and a query.';
      this.errorContainer.classList.add('visible');
      return;
    }
    this.errorContainer.textContent = '';
    const actionData = {
      id: this.editMode ? this.editingId : this._generateId(name),
      name,
      query,
      isEdit: this.editMode
    };
    this.onSubmit(actionData);
    this.hide();
  }

  _generateId(name) {
    return name
      .toLowerCase()
      .replace(/[^\w]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }
}
