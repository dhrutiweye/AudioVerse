# AudioVerse

A Python monorepo project using Poetry Workspaces with ingestion and analytics services.

## Project Structure

```
AudioVerse/
├── pyproject.toml              # Root workspace configuration
├── libs/
│   ├── utils/                  # Shared utilities library
│   │   ├── pyproject.toml
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logger.py       # Logging utilities using loguru
│   └── database/               # Shared database library
│       ├── pyproject.toml
│       └── database/
│           ├── __init__.py
│           └── client.py       # Database client with mock connection
├── apps/
│   ├── ingestion_service/      # Ingestion service app
│   │   ├── pyproject.toml
│   │   └── ingestion_service/
│   │       ├── __init__.py
│   │       └── main.py         # Logs and connects to DB
│   └── analytics_service/      # Analytics service app
│       ├── pyproject.toml
│       └── analytics_service/
│           ├── __init__.py
│           └── main.py         # Logs simple analysis
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   poetry install
   ```

   This will install all dependencies for the workspace and link the shared libraries using Poetry path dependencies.

## Running the Services

### Ingestion Service

```bash
poetry run python apps/ingestion_service/ingestion_service/main.py
```

This service:
- Uses the `utils` library to log messages
- Uses the `database` library to connect to the database
- Logs connection status information

### Analytics Service

```bash
poetry run python apps/analytics_service/analytics_service/main.py
```

This service:
- Uses the `utils` library to log messages
- Performs simple data analysis

## Workspace Configuration

The monorepo uses Poetry workspaces to manage:
- **Shared Libraries**: `libs/utils` and `libs/database`
- **Applications**: `apps/ingestion_service` and `apps/analytics_service`

Shared libraries are linked using Poetry path dependencies with `develop = true`, allowing local development and automatic updates when libraries are modified.

