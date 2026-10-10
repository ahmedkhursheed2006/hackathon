# Screen Privacy Guardian (Medusa_Desktop_Core)

A local desktop application that defends against shoulder-surfing and screen
snooping using the host machine's webcam. Face detection and gaze estimation run
entirely on-device; only a single triggered frame is ever transmitted to a
configurable hosted VLM/TTS provider, and only after explicit user consent.

## Requirements

- Python 3.11+

## Project layout

```
src/medusa/
  core/     data models, enums, classification + debounce/cooldown FSM
  vision/   on-device capture, face detection, proximity, gaze
  server/   local FastAPI WebSocket Status_Server
  shield/   Qt click-through privacy overlay (Desktop_Shield)
  alert/    async VLM -> TTS Alert_Service with pluggable providers
  config/   Configuration_Store, validation, consent, secret redaction
  app/      application bootstrap and three-thread orchestration
tests/      mirrors the package layout
```

## Development

Install the project with its test dependencies and run the suite:

```bash
pip install -e ".[test]"
pytest
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/) with a
minimum of 100 examples per property. Choose a Hypothesis profile via the
`HYPOTHESIS_PROFILE` environment variable (`default`, `dev`, or `thorough`).
