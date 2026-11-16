// Confirmation dialog for forms marked with data-confirm. A submit button may
// override the form's data-confirm-title/-ok/-variant for its own action.
(function () {
  var dialog = document.getElementById("confirm-dialog");
  if (!dialog || typeof dialog.showModal !== "function") return;

  var title = document.getElementById("confirm-dialog-title");
  var message = document.getElementById("confirm-dialog-message");
  var ok = dialog.querySelector("[data-confirm-ok]");
  var cancel = dialog.querySelector("[data-confirm-cancel]");
  var pending = null;

  function setting(form, submitter, name, fallback) {
    return (submitter && submitter.dataset[name]) || form.dataset[name] || fallback;
  }

  document.addEventListener("submit", function (event) {
    var form = event.target;
    if (!form.hasAttribute || !form.hasAttribute("data-confirm")) return;
    if (form.dataset.confirmed === "1") { delete form.dataset.confirmed; return; }

    event.preventDefault();
    var submitter = event.submitter && form.contains(event.submitter) ? event.submitter : null;
    pending = { form: form, submitter: submitter };

    var variant = setting(form, submitter, "confirmVariant", "danger");
    if (["danger", "primary", "zone"].indexOf(variant) === -1) variant = "danger";

    title.textContent = setting(form, submitter, "confirmTitle", "Are you sure?");
    title.classList.toggle("is-danger", variant === "danger");
    message.textContent = form.dataset.confirm || "This cannot be undone.";
    ok.textContent = setting(form, submitter, "confirmOk", "Confirm");
    ok.className = "btn btn-" + variant;
    dialog.showModal();
  });

  ok.addEventListener("click", function () {
    if (!pending) return;
    var form = pending.form;
    var submitter = pending.submitter;
    pending = null;
    dialog.close();
    form.dataset.confirmed = "1";
    try {
      form.requestSubmit(submitter || undefined);
    } catch (error) {
      delete form.dataset.confirmed;
      form.requestSubmit();
      return;
    }
    // Clears the guard if the browser stopped at an invalid field instead of submitting.
    setTimeout(function () { delete form.dataset.confirmed; }, 0);
  });

  function dismiss() {
    pending = null;
    dialog.close();
  }
  cancel.addEventListener("click", dismiss);
  dialog.addEventListener("click", function (event) {
    if (event.target === dialog) dismiss();
  });
  dialog.addEventListener("close", function () { pending = null; });
})();
