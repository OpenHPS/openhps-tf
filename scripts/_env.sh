# Shared .env loader for scripts/. Source this, don't execute it.
#
# Deliberately NOT `set -a && . ./.env`: .env is a config file, not a shell
# script. Sourcing it executes whatever it contains, and it only takes one
# unquoted value to break the caller. So we parse it: only the keys the shell
# scripts actually use are exported, and values are taken literally.
load_env() {
  local file="${1:-.env}"
  [ -f "$file" ] || return 0
  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|'#'*) continue ;;
    esac
    case "$line" in
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    # Trim surrounding whitespace from the key only; values are literal.
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    case "$key" in
      REMOTE_HOST|REMOTE_USER|REMOTE_DIR|REMOTE_IMAGE|REMOTE_GPUS|EPOCHS|MODEL_DIR|BATCH_SIZE|PATIENCE|ENSEMBLE|RUN_NAME) ;;
      *) continue ;;
    esac
    # Strip one layer of matching quotes if present.
    case "$value" in
      \"*\") value="${value#\"}"; value="${value%\"}" ;;
      \'*\') value="${value#\'}"; value="${value%\'}" ;;
    esac
    # Real environment variables win, matching the TypeScript loader.
    if [ -z "$(eval "printf '%s' \"\${$key:-}\"")" ]; then
      export "$key=$value"
    fi
  done < "$file"
}
