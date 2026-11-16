# EC2 deployment

The workflow tests on GitHub-hosted Windows and Ubuntu runners, builds and checks the Docker image on GitHub, then transfers the compressed image over SSH. EC2 loads the finished image instead of installing dependencies or building it locally. This avoids memory-heavy builds on small EC2 instances. It uses your existing `EC2_HOST`, `EC2_USER`, and `EC2_SSH_KEY` repository secrets. The self-hosted runner is not needed for this deployment method.

## One-time server setup (Ubuntu EC2)

SSH into the instance and run:

```bash
sudo apt-get update
sudo apt-get install -y docker.io curl
sudo systemctl enable --now docker
sudo -n docker info
```

The SSH account must have passwordless sudo for the deployment commands. Docker builds the Python 3.14 environment, so Python does not need to be installed on the host. Amazon Linux requires its corresponding Docker installation commands rather than apt-get.

Set these GitHub repository secrets:

- `EC2_HOST`: `3.108.63.240` (no protocol or port).
- `EC2_USER`: the instance SSH username, typically `ubuntu` for Ubuntu or `ec2-user` for Amazon Linux.
- `EC2_SSH_KEY`: the complete private key including BEGIN/END lines. Keep it out of Git.
- Optional `EC2_KNOWN_HOSTS`: verified SSH known_hosts entry for the instance. Without it the job trusts the first presented host key; with it SSH requires the pinned key to match.

Allow inbound TCP 22 from the deployment runner and inbound HTTP TCP 80 from the intended browser clients in the EC2 security group. GitHub-hosted runner addresses vary; an existing restricted SSH rule may prevent the deployment runner from connecting. Ensure port 80 is available: the deployment container binds it directly, so an existing Nginx/Apache listener would conflict. Port 8000 does not need to be opened externally.

## Deploy and verify

Push to `main`, or select **Actions > Mental Health Prediction CI/CD > Run workflow** on `main`. Both test jobs must pass before deployment. Pull requests only run tests. Pushes and manual runs are serialized so an active deployment is not cancelled halfway through.

Open `http://3.108.63.240` and `http://3.108.63.240/health` after the deployment job succeeds. This is HTTP; HTTPS needs a domain and TLS setup separately.

The `mental-health` container restarts after host reboots. Code is stored in `~/mental-health/releases/<commit>` outside the runner workspace. A failed startup restores the previous container. Historical release directories and Docker images are retained for inspection; monitor server disk use.

### RAG configuration

The Knowledge Assistant page is `/rag`. To enable Gemini requests, create `~/mental-health/rag.env` on EC2 with `GEMINI_API_KEY=your_key` (and optional `GEMINI_CHAT_MODEL` / `GEMINI_EMBEDDING_MODEL` overrides), restrict it with `chmod 600 ~/mental-health/rag.env`, then redeploy. The deployment passes this file into Docker without putting the key in Git. Documents are kept in the `mental-health-knowledge` Docker volume across redeployments. This is a shared demo knowledge base: upload educational, non-personal documents only.

Inspect failures on EC2:

```bash
sudo docker ps -a
sudo docker logs --tail 100 mental-health
sudo ss -ltnp | grep ':80 '
curl -f http://127.0.0.1/health
```
