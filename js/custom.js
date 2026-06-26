/* ============================================
   Custom JS for GTilor's Blog
   Premium interactions without modifying theme source
   ============================================ */

(function () {
  "use strict";

  // ---------- TOC 当前章节高亮跟随 (Intersection Observer) ----------
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
          // Remove active from all
          tocLinks.forEach(function (link) {
            link.classList.remove("is-active-link");
          });
          // Add active to matching link
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

  // ---------- 图片加载淡入 ----------
  function initImageFadeIn() {
    var images = document.querySelectorAll("img[loading='lazy'], img[data-src]");
    if (!images.length) return;

    if ("IntersectionObserver" in window) {
      var imgObserver = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              var img = entry.target;
              img.addEventListener("load", function () {
                img.classList.add("loaded");
              });
              // If image is already loaded (cached)
              if (img.complete) {
                img.classList.add("loaded");
              }
              imgObserver.unobserve(img);
            }
          });
        },
        { rootMargin: "100px" }
      );

      images.forEach(function (img) {
        img.classList.add("loaded"); // Fallback: show immediately
        imgObserver.observe(img);
      });
    } else {
      // Fallback for browsers without IntersectionObserver
      images.forEach(function (img) {
        img.classList.add("loaded");
      });
    }
  }

  // ---------- Banner 副标题打字机光标优化 ----------
  function enhanceTypingCursor() {
    var typedCursor = document.querySelector(".typed-cursor");
    if (typedCursor) {
      typedCursor.style.animation = "typedjsBlink 0.7s infinite";
      typedCursor.style.fontWeight = "300";
      typedCursor.style.color = "var(--subtitle-color, #fff)";
    }
  }

  // ---------- Run all enhancements ----------
  document.addEventListener("DOMContentLoaded", function () {
    // TOC observer - wait a bit for tocbot to finish rendering
    setTimeout(initTocObserver, 500);

    // Image fade-in
    initImageFadeIn();

    // Typing cursor enhancement
    enhanceTypingCursor();
  });

  // Re-run TOC observer after potential dynamic content changes
  window.addEventListener("load", function () {
    setTimeout(initTocObserver, 300);
  });
})();
