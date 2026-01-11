# Multi-stage build for alpha-toe teach site
FROM node:20-slim AS node-base

# Install Python and uv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Build stage
FROM node-base AS builder

WORKDIR /app/teach

# Install Node dependencies
RUN npm install

# Run build_pages.py to generate HTML from notebooks
RUN uv run python build_pages.py

# Build the site with Vite
RUN npm run build

# Production stage
FROM node-base AS production

WORKDIR /app/teach/dist

# Copy built files from builder
COPY --from=builder /app/teach/dist ./

# Expose port 1122 for Cloudflare tunnel
EXPOSE 1122

# Serve the static files
CMD ["uv", "run", "python", "-m", "http.server", "1122"]
