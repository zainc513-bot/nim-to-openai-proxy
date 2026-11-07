// utils.js — small shared helpers

'use strict';

/**
 * Length of the longest suffix of `str` that is a proper prefix of `tag`.
 * Used by streaming parsers to hold back text that might be the start of a
 * tag split across two chunks.
 */
function partialLen(str, tag) {
  for (let i = Math.min(str.length, tag.length - 1); i > 0; i--) {
    if (tag.startsWith(str.slice(-i))) return i;
  }
  return 0;
}

module.exports = { partialLen };
