# Adding a new model in 5 steps

1. Copy this folder: `cp -r models/_template models/<model-name>`
2. Fill in the Dockerfile: engine source + pinned SHA + build command + worker file.
3. Write the worker: two routes are enough — `POST /tts` for audio and `GET /health` for health.
4. Add a service with a different port in `docker-compose.yml`.
5. Build and test: `docker compose build` then `curl localhost:<PORT>/health`.

Rule: every model has its own folder; other models' files stay untouched.
