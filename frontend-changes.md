# Frontend Changes

## Code Quality Tooling

### What was added

**Prettier** — automatic code formatter for HTML, CSS, and JavaScript (the frontend equivalent of Black for Python).

### New files

| File | Purpose |
|---|---|
| `.prettierrc` | Prettier configuration (2-space indent, single quotes, LF line endings, 80-char print width) |
| `package.json` | Declares Prettier as a dev dependency; provides `npm run format` and `npm run format:check` scripts |
| `scripts/check-frontend.sh` | Shell script to run Prettier checks from the project root; pass `--fix` to auto-format |

### Modified files

All three frontend source files were reformatted to match Prettier output:

- `frontend/index.html` — 4-space → 2-space indent; Prettier-standard attribute layout for multi-attribute elements
- `frontend/style.css` — 4-space → 2-space indent; each selector on its own line for grouped rules (e.g. `h1`, `h2`, `h3`); `@keyframes` stops expanded to multi-line
- `frontend/script.js` — 4-space → 2-space indent; double quotes → single quotes; trailing commas added in object/array literals; method chains broken across lines at the 80-char boundary; removed stray blank lines

### How to use

```bash
# Install dev dependencies (one-time)
npm install

# Check formatting
npm run format:check
# or
./scripts/check-frontend.sh

# Auto-fix formatting
npm run format
# or
./scripts/check-frontend.sh --fix
```

No Node.js is required to run the application itself — only to run formatting checks.
