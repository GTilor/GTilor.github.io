/* global hexo */
'use strict';

hexo.extend.filter.register('markdown-it:renderer', function(md) {
  md.use(require('@traptitech/markdown-it-katex'));
});
