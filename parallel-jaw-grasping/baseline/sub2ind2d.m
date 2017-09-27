% A faster version of sub2ind for 2D case
function linIndex = sub2ind2d(sz, rowSub, colSub)
  linIndex = (colSub-1) * sz(1) + rowSub;

