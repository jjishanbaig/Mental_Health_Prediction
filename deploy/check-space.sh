#!/usr/bin/env bash
set -euo pipefail
# Run before uploading the compressed image. Leave room for the archive,
# unpacked image and the previous version needed for rollback.
minimum_kb=6291456
for path in "$HOME" /var/lib/docker /var/lib/containerd; do
  if [ -d "$path" ]; then
    available=$(sudo -n df -Pk "$path" | awk 'NR==2 {print $4}')
    if [ "$available" -lt "$minimum_kb" ]; then
      echo "Insufficient free disk space on $path: need at least 6 GiB before deployment."
      sudo -n df -h "$path"
      echo 'Free unused build cache or expand the EC2 volume, then rerun deployment. Do not delete knowledge volumes.'
      exit 1
    fi
  fi
done
