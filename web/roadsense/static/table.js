/* Shared client-side filter, search, and pagination for RoadSense tables. */
window.initDataTable = function (cfg) {
  var search = document.getElementById(cfg.searchId);
  var tabs = document.querySelectorAll(cfg.tabSelector || ".users-tab");
  var tableWrap = document.getElementById(cfg.wrapId);
  var emptyEl = document.getElementById(cfg.emptyId);
  var pagerEl = document.getElementById(cfg.pagerId);
  var rows = Array.prototype.slice.call(document.querySelectorAll("#" + cfg.tableId + " tbody tr"));
  var pageSize = cfg.pageSize || 10;
  var activeFilter = "all";
  var currentPage = 1;

  function getMatchedRows() {
    var q = (search && search.value || "").trim().toLowerCase();
    return rows.filter(function (tr) {
      var role = tr.getAttribute("data-role") || "";
      var status = tr.getAttribute("data-status") || "";
      var hay = tr.getAttribute("data-search") || "";
      var matchTab =
        activeFilter === "all" ||
        status === activeFilter ||
        role === activeFilter;
      var matchQuery = !q || hay.indexOf(q) !== -1;
      return matchTab && matchQuery;
    });
  }

  function renderPager(total, totalPages) {
    if (!pagerEl) return;
    if (total === 0 || totalPages <= 1) {
      pagerEl.style.display = total === 0 ? "none" : "flex";
      if (total === 0) {
        pagerEl.innerHTML = "";
        return;
      }
    }
    pagerEl.style.display = "flex";
    var start = (currentPage - 1) * pageSize + 1;
    var end = Math.min(total, currentPage * pageSize);

    var html = '<span class="pager-info">Showing <strong>' + start + '&ndash;' + end + '</strong> of <strong>' + total + '</strong></span>';
    if (totalPages > 1) {
      html += '<div class="pager-btns">';
      html += '<button type="button" class="btn btn-outline btn-sm" data-page="' + (currentPage - 1) + '"' + (currentPage === 1 ? ' disabled' : '') + '>Previous</button>';
      for (var p = 1; p <= totalPages; p++) {
        if (p === 1 || p === totalPages || Math.abs(p - currentPage) <= 1) {
          html += '<button type="button" class="pager-num' + (p === currentPage ? ' is-active' : '') + '" data-page="' + p + '">' + p + '</button>';
        } else if (Math.abs(p - currentPage) === 2) {
          html += '<span class="pager-ellipsis">&hellip;</span>';
        }
      }
      html += '<button type="button" class="btn btn-outline btn-sm" data-page="' + (currentPage + 1) + '"' + (currentPage === totalPages ? ' disabled' : '') + '>Next</button>';
      html += '</div>';
    }
    pagerEl.innerHTML = html;

    pagerEl.querySelectorAll("button[data-page]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var next = parseInt(btn.getAttribute("data-page"), 10);
        if (!isNaN(next) && next >= 1 && next <= totalPages) {
          currentPage = next;
          render();
        }
      });
    });
  }

  function render() {
    var matched = getMatchedRows();
    var total = matched.length;
    var totalPages = Math.max(1, Math.ceil(total / pageSize));
    if (currentPage > totalPages) currentPage = totalPages;

    var startIdx = (currentPage - 1) * pageSize;
    var endIdx = startIdx + pageSize;
    var visibleSet = new Set(matched.slice(startIdx, endIdx));

    rows.forEach(function (tr) {
      tr.style.display = visibleSet.has(tr) ? "" : "none";
    });

    if (tableWrap) tableWrap.style.display = total ? "" : "none";
    if (emptyEl) emptyEl.style.display = total ? "none" : "";
    renderPager(total, totalPages);
  }

  if (search) {
    search.addEventListener("input", function () {
      currentPage = 1;
      render();
    });
  }

  tabs.forEach(function (btn) {
    btn.addEventListener("click", function () {
      tabs.forEach(function (b) { b.classList.remove("is-active"); });
      btn.classList.add("is-active");
      activeFilter = btn.getAttribute("data-filter") || "all";
      currentPage = 1;
      render();
    });
  });

  render();
};
