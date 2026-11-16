#!/usr/bin/env bash
# Executed over SSH by CI. Requires Docker and passwordless sudo on EC2.
set -euo pipefail
revision="${1:?Missing commit SHA}"
[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo 'Invalid commit SHA'; exit 1; }
command -v docker >/dev/null || { echo 'Install and start Docker on EC2 first; see deploy/README.md'; exit 1; }
sudo -n docker info >/dev/null

release="$HOME/mental-health/releases/$revision"
mkdir -p "$release"
tar -xzf "$HOME/mental-health-$revision.tar.gz" -C "$release"
cd "$release"
image_archive="$HOME/mental-health-image-$revision.tar.gz"
test -f "$image_archive" || { echo 'Prebuilt image archive is missing. Run the updated CI/CD workflow.'; exit 1; }
echo 'Loading prebuilt image; dependency installation runs on GitHub, not EC2.'
sudo -n docker load --input "$image_archive"
sudo -n docker image inspect "mental-health:$revision" > /dev/null
rm -f "$image_archive"
# docker load can register image metadata even when lazy layer extraction fails.
# Force extraction and validate the runtime while the old app is still running.
sudo -n docker run --rm --entrypoint python "mental-health:$revision" \
  -c "from backend.app.rag_service import guide_search_answer; assert guide_search_answer('What is stress?')['sources']"

name=mental-health
backup=mental-health-previous
if sudo -n docker container inspect "$backup" >/dev/null 2>&1; then
  echo 'An earlier rollback container exists. Inspect mental-health-previous before retrying.'
  exit 1
fi
had_previous=false
if sudo -n docker container inspect "$name" >/dev/null 2>&1; then
  sudo -n docker stop "$name"
  sudo -n docker rename "$name" "$backup"
  had_previous=true
fi
rollback() {
  echo 'Deployment failed; restoring previous container when available.'
  sudo -n docker logs --tail 60 "$name" || true
  sudo -n docker rm -f "$name" || true
  if "$had_previous"; then
    sudo -n docker rename "$backup" "$name"
    sudo -n docker start "$name"
  fi
}
trap rollback ERR
rag_environment=()
if [ -f "$HOME/mental-health/rag.env" ]; then
  rag_environment=(--env-file "$HOME/mental-health/rag.env")
fi
sudo -n docker run --detach --name "$name" --restart unless-stopped \
  "${rag_environment[@]}" \
  --mount type=volume,source=mental-health-knowledge,target=/app/backend/knowledge_base \
  --publish 80:8000 --log-opt max-size=10m --log-opt max-file=3 "mental-health:$revision"
healthy=false
for attempt in $(seq 1 30); do
  if sudo -n docker exec "$name" python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)" >/dev/null 2>&1; then
    healthy=true
    break
  fi
  sleep 2
done
"$healthy" || { echo 'Application health check failed'; false; }
# Verify the published port serves the frontend, not just the internal API.
curl --fail --silent --show-error http://127.0.0.1/ > /dev/null
trap - ERR
if "$had_previous"; then sudo -n docker rm "$backup"; fi
rm -f "$HOME/mental-health-$revision.tar.gz"
echo "Successfully deployed $revision on port 80"
