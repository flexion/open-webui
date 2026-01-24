# FlexChat - Flexion's Open WebUI Fork

FlexChat is Flexion's customized deployment of [Open WebUI](https://github.com/open-webui/open-webui), rebranded and configured to integrate with AWS Bedrock via the [Flexion Bedrock Access Gateway](https://github.com/flexion/bedrock-access-gateway).

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `flex` | **Flexion customizations** - Contains all Flexion-specific branding, configurations, and features. This is the primary branch for Flexion development. |
| `main` | Mirrors the upstream Open WebUI `main` branch. Used for tracking upstream releases. |
| `dev` | Mirrors the upstream Open WebUI `dev` branch. Used for tracking upstream development. |

### Keeping Up with Upstream

To incorporate upstream Open WebUI updates into our `flex` branch:

```bash
# Add upstream remote (one-time setup)
git remote add upstream https://github.com/open-webui/open-webui.git
```

## Local Development Setup

### Prerequisites

- Docker and Docker Compose
- AWS credentials configured (for Bedrock Access Gateway)
- [Flexion Bedrock Access Gateway](https://github.com/flexion/bedrock-access-gateway) running locally (or live connection)

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│                 │     │                         │     │                 │
│    FlexChat     │────▶│  Bedrock Access Gateway │────▶│  AWS Bedrock    │
│   (Port 3000)   │     │      (Port 8000)        │     │                 │
│                 │     │                         │     │                 │
└─────────────────┘     └─────────────────────────┘     └─────────────────┘
```

FlexChat connects to the Bedrock Access Gateway (BAG) which provides an OpenAI-compatible API facade for Amazon Bedrock models.

### Step 1: Start the Bedrock Access Gateway

Before running FlexChat, you need the Bedrock Access Gateway running locally. See the [BAG README_FLEXION.md](https://github.com/flexion/bedrock-access-gateway/blob/main/README_FLEXION.md) for detailed setup instructions.

Quick start for BAG:

```bash
# Clone the Bedrock Access Gateway repo
git clone https://github.com/flexion/bedrock-access-gateway.git
cd bedrock-access-gateway

# Set up virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

# Configure AWS and start the gateway
export AWS_REGION=us-east-1
export API_KEY=bedrock
export ALLOWED_MODEL_IDS='["anthropic.*", "us.anthropic.*", "us.meta.*"]'

# Run on port 8000
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Step 2: Configure FlexChat Environment

Create or update your `.env` file in the FlexChat root directory:

```bash
# Core Settings
ENV=dev
WEBUI_AUTH=FALSE
ENABLE_LOGIN_FORM=false

# Model Configuration
BYPASS_MODEL_ACCESS_CONTROL=true
DEFAULT_MODELS=us.meta.llama3-1-8b-instruct-v1:0

# Bedrock Access Gateway Connection
# Points to the locally running BAG instance
OPENAI_API_BASE_URL=http://host.docker.interal:8000/api/v1
OPENAI_API_KEY=bedrock

# Disable Ollama (we're using Bedrock)
ENABLE_OLLAMA_API=false

# User Settings
DEFAULT_USER_ROLE=user
ENABLE_API_KEYS=true
USER_PERMISSIONS_FEATURES_API_KEYS=true
```

### Step 3: Run FlexChat with Docker Compose

```bash
# Build and start FlexChat
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

FlexChat will be available at `http://localhost:3000`.

### Docker Compose Configuration

The `docker-compose.yaml` is configured to:

- Build FlexChat from the local Dockerfile
- Mount a persistent volume for data storage
- Load environment variables from `.env`
- Expose local port services like the BAG
- Add `host.docker.internal` for accessing host services (like the BAG)

```yaml
services:
  open-webui:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: open-webui
    volumes:
      - open-webui:/app/backend/data
    network_mode: 'host'
    env_file:
      - .env
    extra_hosts:
      - host.docker.internal:host-gateway
    restart: unless-stopped
```

**Note:** The Ollama service is included in the compose file but is not required when using Bedrock. You can remove or comment out the Ollama service and its dependency if desired.

## Connecting to Bedrock Access Gateway

The `OPENAI_API_BASE_URL` environment variable is set to `http://host.docker.internal:8000/api/v1` to connect FlexChat to the locally running Bedrock Access Gateway.

### Important Notes

- **API Key:** The `OPENAI_API_KEY=bedrock` matches the default API key used by the BAG in local development mode.
- **Available Models:** The models available in FlexChat depend on the `ALLOWED_MODEL_IDS` configured in the Bedrock Access Gateway.

### Updating OPENAI_API_BASE_URL for Docker

If FlexChat is running in Docker and BAG is running on your host:

```bash
# In .env, use host.docker.internal for Docker-to-host communication
OPENAI_API_BASE_URL=http://host.docker.internal:8000/api/v1
```

## Flexion Customizations

The `flex` branch includes the following Flexion-specific changes:

### Branding
- Application name changed from "Open WebUI" to "FlexChat"
- Custom Flexion logo used for favicons and splash screens
- Updated site manifest and HTML title

### Configuration Defaults
- Default integration with Bedrock Access Gateway
- Ollama disabled by default
- Google OAuth pre-configured (credentials required)

## Troubleshooting

### FlexChat can't connect to models

1. Verify the Bedrock Access Gateway is running on port 8000
2. Check that `OPENAI_API_BASE_URL` is correctly set
3. If using Docker, try using `0.0.0.0` and setting `network_mode: 'host'`

### No models appearing in the UI

1. Check BAG logs for any authentication errors
2. Verify your AWS credentials have Bedrock invoke permissions
3. Confirm `ALLOWED_MODEL_IDS` in BAG includes the models you expect

### Docker build fails

1. Ensure Docker has sufficient resources allocated
2. Try clearing Docker cache: `docker-compose build --no-cache`

## Related Documentation

- [Open WebUI Documentation](https://docs.openwebui.com/)
- [Flexion Bedrock Access Gateway](https://github.com/flexion/bedrock-access-gateway/blob/main/README_FLEXION.md)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
