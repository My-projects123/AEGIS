#!/bin/sh
set -eu
mkdir -p public/static public/api
cp frontend/index.html public/index.html
cp frontend/styles.css frontend/app.js frontend/aegis.js public/static/
node -e "
const m = require('./ml/models/tiny_mask.meta.json');
m.available = false;
m.hosted_static = true;
m.backend = 'none (static host)';
require('fs').writeFileSync('public/api/ml', JSON.stringify(m));
"
