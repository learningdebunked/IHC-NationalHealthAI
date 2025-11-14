.PHONY: help install setup dev test clean docker-up docker-down lint format

# Default target
help:
	@echo "IHC Platform - Available Commands:"
	@echo ""
	@echo "  make install       - Install all dependencies"
	@echo "  make setup         - Initial project setup"
	@echo "  make dev           - Start development servers"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make clean         - Clean generated files"
	@echo ""

# Install dependencies
install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Installing ML dependencies..."
	cd ml && pip install -r requirements.txt
	@echo "✅ All dependencies installed"

# Initial setup
setup: install
	@echo "Setting up environment files..."
	cp backend/.env.example backend/.env
	cp frontend/.env.example frontend/.env
	@echo "Creating directories..."
	mkdir -p logs ml/models/saved
	@echo "✅ Setup complete"

# Start development servers
dev:
	@echo "Starting development servers..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "API Docs: http://localhost:8000/api/v1/docs"
	@make -j2 dev-backend dev-frontend

dev-backend:
	cd backend && uvicorn main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# Run tests
test:
	@echo "Running backend tests..."
	cd backend && pytest tests/ -v --cov=. --cov-report=html
	@echo "Running frontend tests..."
	cd frontend && npm test
	@echo "✅ All tests passed"

# Run linters
lint:
	@echo "Linting backend code..."
	cd backend && flake8 . --max-line-length=100
	cd backend && mypy . --ignore-missing-imports
	@echo "Linting frontend code..."
	cd frontend && npm run lint
	@echo "✅ Linting complete"

# Format code
format:
	@echo "Formatting backend code..."
	cd backend && black . --line-length=100
	cd backend && isort .
	@echo "Formatting frontend code..."
	cd frontend && npm run format
	@echo "✅ Code formatted"

# Docker commands
docker-up:
	@echo "Starting Docker containers..."
	docker-compose up -d
	@echo "✅ Containers started"
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "Grafana: http://localhost:3001"
	@echo "Prometheus: http://localhost:9090"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✅ Containers stopped"

docker-logs:
	docker-compose logs -f

docker-rebuild:
	@echo "Rebuilding Docker containers..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Containers rebuilt"

# Database commands
db-migrate:
	@echo "Running database migrations..."
	cd backend && alembic upgrade head
	@echo "✅ Migrations complete"

db-reset:
	@echo "Resetting database..."
	cd backend && alembic downgrade base
	cd backend && alembic upgrade head
	@echo "✅ Database reset"

# ML model training
train-eligibility:
	@echo "Training eligibility classifier..."
	cd ml && python training/train_eligibility_classifier.py
	@echo "✅ Training complete"

train-spending:
	@echo "Training spending predictor..."
	cd ml && python training/train_spending_model.py
	@echo "✅ Training complete"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf backend/logs/* frontend/.next frontend/out
	@echo "✅ Cleanup complete"

# Production build
build:
	@echo "Building for production..."
	cd frontend && npm run build
	@echo "✅ Build complete"

# Deploy
deploy:
	@echo "Deploying to production..."
	@echo "⚠️  Make sure you have configured your deployment settings"
	# Add your deployment commands here
	@echo "✅ Deployment complete"