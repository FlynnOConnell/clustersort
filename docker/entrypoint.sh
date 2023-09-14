#!/bin/bash --login
set -econda activate $ENV_PREFIX
exec "$@"