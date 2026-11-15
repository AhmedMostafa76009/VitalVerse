/* Vitalverse shared navbar/footer/modal/security shell helper */
(function () {
  'use strict';

  function esc(str) {
    return String(str == null ? '' : str).replace(/[&<>"']/g, function (c) {
      return {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }[c];
    });
  }

  function safeUrl(url) {
    if (!url) return '';
    try {
      var u = new URL(String(url), window.location.origin);
      return (u.protocol === 'https:' || u.protocol === 'http:') ? u.href : '';
    } catch (e) {
      return '';
    }
  }

  var state = {
    authenticated: false,
    user: null,
    role: null,
    csrfToken: null,
    todoPercent: 0,
    unreadAlerts: 0,
    unackDanger: 0,
    alerts: []
  };

  function currentPage() {
    var path = (window.location.pathname || '/').split('/').pop() || 'index.html';
    return path.toLowerCase();
  }

  function initActiveMenu() {
    var page = currentPage();
    var map = {
      '': 'index.html',
      'index.html': 'index.html',
      'health.html': 'health.html',
      'astronaut-dashboard.html': 'health.html',
      'doctor-dashboard.html': 'health.html',
      'chatbot.html': 'chatbot.html',
      'mental.html': 'mental.html',
      'mental_astronaut.html': 'mental.html',
      'mental_doctor.html': 'mental.html',
      'about.html': 'about.html'
    };
    var target = map[page] || '';
    document.querySelectorAll('#vvNavMenu a').forEach(function (a) {
      var href = (a.getAttribute('href') || '').toLowerCase();
      if (target && href === target) {
        a.classList.add('active');
      } else {
        a.classList.remove('active');
      }
    });
  }

  function setTodoProgress(pct) {
    var ring = document.getElementById('vvTodoRing');
    var circle = document.getElementById('vvTodoRingArc');
    var clamped = Math.max(0, Math.min(100, Number(pct) || 0));
    state.todoPercent = clamped;
    if (!ring || !circle) return;
    if (clamped <= 0) {
      ring.setAttribute('hidden', '');
      return;
    }
    var r = 18;
    var c = 2 * Math.PI * r;
    circle.style.strokeDasharray = c.toFixed(2);
    circle.style.strokeDashoffset = (c * (1 - clamped / 100)).toFixed(2);
    ring.removeAttribute('hidden');
  }

  function updateBell(unread, unackDanger) {
    var dot = document.getElementById('vvBellDot');
    state.unreadAlerts = Number(unread) || 0;
    state.unackDanger = Number(unackDanger) || 0;
    if (!dot) return;
    if (state.unreadAlerts <= 0 && state.unackDanger <= 0) {
      dot.setAttribute('hidden', '');
      dot.classList.remove('is-danger');
      return;
    }
    dot.removeAttribute('hidden');
    if (state.unackDanger > 0) {
      dot.classList.add('is-danger');
    } else {
      dot.classList.remove('is-danger');
    }
  }

  function ensureSubpageWelcomeModal() {
    var existing = document.getElementById('welcomeOverlay') || document.getElementById('vvWelcomeOverlay');
    if (existing) return existing;

    var wrap = document.createElement('div');
    wrap.id = 'vvWelcomeOverlay';
    wrap.className = 'vv-modal-overlay';
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-modal', 'true');
    wrap.setAttribute('aria-labelledby', 'vvWelcomeTitle');
    wrap.innerHTML =
      '<div class="vv-modal">' +
        '<button class="vv-modal-close" type="button" aria-label="Close modal">&times;</button>' +
        '<h2 id="vvWelcomeTitle">Welcome to <span>Vitalverse</span></h2>' +
        '<p>Sign in with your astronaut or flight-surgeon account, or create a new mission profile.</p>' +
        '<div class="vv-modal-actions">' +
          '<a class="vv-modal-primary" href="signup.html">Create an Account</a>' +
          '<a class="vv-modal-secondary" href="login.html">Log In</a>' +
        '</div>' +
      '</div>';
    document.body.appendChild(wrap);

    var closeBtn = wrap.querySelector('.vv-modal-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', function () {
        wrap.classList.remove('show');
      });
    }
    wrap.addEventListener('click', function (e) {
      if (e.target === wrap) wrap.classList.remove('show');
    });
    return wrap;
  }

  function openWelcomeModal() {
    var modal = ensureSubpageWelcomeModal();
    if (modal) modal.classList.add('show');
  }

  function closeAllMenus() {
    document.querySelectorAll('.vv-dropdown').forEach(function (d) {
      d.setAttribute('hidden', '');
    });
    var profBtn = document.getElementById('profileBtn');
    var bellBtn = document.getElementById('bellBtn');
    if (profBtn) profBtn.setAttribute('aria-expanded', 'false');
    if (bellBtn) bellBtn.setAttribute('aria-expanded', 'false');
  }

  function wireControls() {
    var todoBtn = document.getElementById('todoBtn');
    if (todoBtn && todoBtn.tagName !== 'A') {
      todoBtn.addEventListener('click', function () {
        window.location.href = 'todo.html';
      });
    }

    var bellBtn = document.getElementById('bellBtn');
    var bellDrop = document.getElementById('vvBellDropdown');
    if (bellBtn) {
      bellBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        if (!bellDrop) {
          window.location.href = 'alerts.html';
          return;
        }
        var isOpen = !bellDrop.hasAttribute('hidden');
        closeAllMenus();
        if (!isOpen) {
          bellDrop.removeAttribute('hidden');
          bellBtn.setAttribute('aria-expanded', 'true');
        }
      });
    }

    var profileBtn = document.getElementById('profileBtn');
    var profDrop = document.getElementById('vvProfileDropdown');
    if (profileBtn) {
      profileBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        if (!state.authenticated || !profDrop) {
          closeAllMenus();
          openWelcomeModal();
          return;
        }
        var isOpen = !profDrop.hasAttribute('hidden');
        closeAllMenus();
        if (!isOpen) {
          profDrop.removeAttribute('hidden');
          profileBtn.setAttribute('aria-expanded', 'true');
        }
      });
    }

    var homeOverlay = document.getElementById('welcomeOverlay');
    if (homeOverlay) {
      ['closeModal', 'closeWelcome'].forEach(function (cid) {
        var closeHome = document.getElementById(cid);
        if (closeHome) {
          closeHome.addEventListener('click', function () {
            homeOverlay.classList.remove('show');
          });
        }
      });
      homeOverlay.addEventListener('click', function (e) {
        if (e.target === homeOverlay) homeOverlay.classList.remove('show');
      });
    }

    var launchBtn = document.getElementById('launchMissionBtn');
    if (launchBtn) {
      launchBtn.addEventListener('click', function (e) {
        e.preventDefault();
        window.location.href = state.authenticated ? 'health.html' : 'login.html';
      });
    }

    document.addEventListener('click', function (e) {
      if (!e.target.closest || !e.target.closest('.tools')) {
        closeAllMenus();
      }
    });

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        closeAllMenus();
        var ov1 = document.getElementById('welcomeOverlay');
        var ov2 = document.getElementById('vvWelcomeOverlay');
        if (ov1) ov1.classList.remove('show');
        if (ov2) ov2.classList.remove('show');
      }
    });
  }

  function renderProfileMenu(data) {
    var avatar = document.getElementById('vvAvatar');
    var profDrop = document.getElementById('vvProfileDropdown');
    var u = (data && data.user) || {};
    if (avatar) {
      avatar.textContent = u.initials || 'VV';
    }
    if (!profDrop) return;

    var roleLabel = data.role === 'doctor' ? 'Doctor / Flight Surgeon' : 'Astronaut';
    var subMeta = u.astronaut_id ? esc(u.vitalverse_id + ' · ' + u.astronaut_id) : esc(u.vitalverse_id || '');
    profDrop.innerHTML =
      '<div class="vv-dropdown-head" style="flex-direction:column;align-items:flex-start;gap:4px;">' +
        '<strong>' + esc(u.name || u.vitalverse_id || 'Crew Member') + '</strong>' +
        '<span style="font-size:11px;color:#a8e07a;font-weight:600;">' + subMeta + '</span>' +
        '<span style="font-size:11px;color:#9ba5ab;">' + esc(roleLabel) + '</span>' +
      '</div>' +
      '<a class="vv-profile-item" href="/health.html">Health dashboard</a>' +
      '<a class="vv-profile-item" href="/chatbot.html">Ai chatbot</a>' +
      '<a class="vv-profile-item" href="/mental.html">Mental Health</a>' +
      '<button type="button" class="vv-profile-item vv-profile-item--danger" id="vvLogoutBtn">Log out</button>';

    var logoutBtn = document.getElementById('vvLogoutBtn');
    if (logoutBtn) {
      logoutBtn.addEventListener('click', function () {
        var headers = {};
        if (state.csrfToken) headers['X-CSRF-Token'] = state.csrfToken;
        fetch('/api/auth/logout', {
          method: 'POST',
          headers: headers,
          credentials: 'same-origin'
        }).finally(function () {
          state.authenticated = false;
          state.user = null;
          state.role = null;
          state.csrfToken = null;
          window.location.href = '/login.html';
        });
      });
    }
  }

  var _sessionPromise = null;

  function hydrateSessionIfAvailable() {
    if (window.location.protocol === 'file:' || typeof window.fetch !== 'function') {
      return Promise.resolve({ authenticated: false });
    }
    if (_sessionPromise) return _sessionPromise;

    _sessionPromise = fetch('/api/auth/me', { credentials: 'same-origin' })
      .then(function (res) {
        if (!res.ok) return { authenticated: false };
        return res.json();
      })
      .then(function (data) {
        if (data && data.authenticated) {
          state.authenticated = true;
          state.user = data.user || null;
          state.role = data.role || null;
          state.csrfToken = data.csrf_token || null;
          setTodoProgress(data.todo_percent || 0);
          updateBell(data.unread_alert_count || 0, data.unack_danger_count || 0);
          renderProfileMenu(data);
        }
        document.dispatchEvent(new CustomEvent('vv:session', { detail: data || { authenticated: false } }));
        return data || { authenticated: false };
      })
      .catch(function () {
        var fallback = { authenticated: false };
        document.dispatchEvent(new CustomEvent('vv:session', { detail: fallback }));
        return fallback;
      });
    return _sessionPromise;
  }

  window.VV_SHELL = {
    esc: esc,
    safeUrl: safeUrl,
    state: state,
    setTodoProgress: setTodoProgress,
    updateBell: updateBell,
    openWelcomeModal: openWelcomeModal,
    getSession: hydrateSessionIfAvailable,
    init: function () {
      initActiveMenu();
      wireControls();
      setTodoProgress(0);
      updateBell(0, 0);
      hydrateSessionIfAvailable();
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', window.VV_SHELL.init);
  } else {
    window.VV_SHELL.init();
  }
})();
