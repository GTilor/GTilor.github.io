/* ============================================
   Custom JS for GTilor's Blog
   Minimal enhancements for academic style
   ============================================ */

(function () {
  "use strict";

  // ---------- TOC active section tracking (Intersection Observer) ----------
  function initTocObserver() {
    var tocLinks = document.querySelectorAll(".tocbot-link");
    if (!tocLinks.length) return;

    var headings = document.querySelectorAll(
      ".post-content h1, .post-content h2, .post-content h3, .post-content h4, .post-content h5, .post-content h6"
    );
    if (!headings.length) return;

    var observerOptions = {
      rootMargin: "-80px 0px -70% 0px",
      threshold: 0,
    };

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var id = entry.target.getAttribute("id");
          if (!id) return;
          tocLinks.forEach(function (link) {
            link.classList.remove("is-active-link");
          });
          var activeLink = document.querySelector('.tocbot-link[href="#' + CSS.escape(id) + '"]');
          if (activeLink) {
            activeLink.classList.add("is-active-link");
          }
        }
      });
    }, observerOptions);

    headings.forEach(function (heading) {
      observer.observe(heading);
    });
  }

  // ---------- Image fade-in on load ----------
  function initImageFadeIn() {
    var images = document.querySelectorAll("img[loading='lazy'], img[data-src]");
    if (!images.length) return;

    images.forEach(function (img) {
      if (img.complete) {
        img.style.opacity = "1";
      } else {
        img.style.opacity = "0";
        img.addEventListener("load", function () {
          img.style.opacity = "1";
        });
      }
    });
  }

  // ---------- Initialize ----------
  document.addEventListener("DOMContentLoaded", function () {
    setTimeout(initTocObserver, 500);
    initImageFadeIn();
  });

  window.addEventListener("load", function () {
    setTimeout(initTocObserver, 300);
  });
})();
