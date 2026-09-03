(function () {
  'use strict';

  /* ================= Language toggle (EN default, persisted) ================= */
  var html = document.documentElement;
  var langBtn = document.getElementById('langToggle');
  var LANG_KEY = 'site-lang';

  function applyLang(lang) {
    html.setAttribute('data-lang', lang);
    html.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
    langBtn.textContent = lang === 'zh' ? 'EN' : '中文';
  }

  var saved = null;
  try { saved = localStorage.getItem(LANG_KEY); } catch (e) { /* private mode */ }
  applyLang(saved === 'zh' ? 'zh' : 'en');

  langBtn.addEventListener('click', function () {
    var next = html.getAttribute('data-lang') === 'zh' ? 'en' : 'zh';
    applyLang(next);
    try { localStorage.setItem(LANG_KEY, next); } catch (e) { /* private mode */ }
  });

  /* ================= Mobile nav ================= */
  var navToggle = document.getElementById('navToggle');
  var navLinks = document.getElementById('navLinks');

  navToggle.addEventListener('click', function () {
    var open = navLinks.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  });

  // Close the menu after choosing a destination
  navLinks.addEventListener('click', function (e) {
    if (e.target.closest('a')) {
      navLinks.classList.remove('open');
      navToggle.setAttribute('aria-expanded', 'false');
    }
  });

  /* ================= Scroll-spy (active nav link) =================
     Active = the last nav section whose top has crossed 35% of the
     viewport. The footer can never quite reach that line (page-end
     clamping), so force it active once we're at the very bottom. */
  var navAnchors = Array.prototype.slice.call(document.querySelectorAll('[data-nav]'));
  var spyTargets = navAnchors
    .map(function (a) { return document.querySelector(a.getAttribute('href')); })
    .filter(Boolean)
    .sort(function (a, b) { return a.offsetTop - b.offsetTop; }); // document order

  var spyMap = {};
  navAnchors.forEach(function (a) {
    spyMap[a.getAttribute('href').slice(1)] = a;
  });

  function updateSpy() {
    var atBottom = window.scrollY >=
      document.documentElement.scrollHeight - window.innerHeight - 2;
    var line = window.innerHeight * 0.35;
    var current = null;
    spyTargets.forEach(function (s) {
      if (atBottom || s.getBoundingClientRect().top <= line) current = s.id;
    });
    Object.keys(spyMap).forEach(function (id) {
      spyMap[id].classList.toggle('active', id === current);
    });
  }

  // Direct call on scroll — the page is small, so this stays cheap and avoids
  // any rAF-throttling dead time (e.g. backgrounded/headless tabs).
  window.addEventListener('scroll', updateSpy, { passive: true });
  window.addEventListener('resize', updateSpy, { passive: true });
  updateSpy();

  /* ================= Scroll-reveal (subtle fade-up) ================= */
  var revealEls = document.querySelectorAll(
    '.section, .project-card, .note-group, .skill-card, .award-card, .stat'
  );

  if ('IntersectionObserver' in window) {
    var reveal = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          reveal.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    revealEls.forEach(function (el) {
      el.classList.add('reveal');
      reveal.observe(el);
    });
  }
})();
