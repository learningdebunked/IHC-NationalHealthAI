#!/bin/bash

# Intelligent Health Checkout (IHC) Platform - Workspace Setup Script
# This script creates the complete directory structure and placeholder files

set -e

echo "🏗️  Setting up IHC Platform workspace..."

# Create main directories
mkdir -p backend/{api,models/{eligibility,spending,assistant},services,database,utils,config}
mkdir -p frontend/{src/{components,pages,services,utils,hooks,contexts,types},public}
mkdir -p ml/{notebooks,training,evaluation,data/{raw,processed,external},configs}
mkdir -p deployment/{docker,kubernetes,terraform}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docs/{api,architecture,deployment,user-guide}
mkdir -p scripts

echo "📁 Directory structure created"

# Backend files
touch backend/__init__.py
touch backend/main.py
touch backend/requirements.txt
touch backend/config.py
touch backend/.env.example

# Backend API
touch backend/api/__init__.py
touch backend/api/routes.py
touch backend/api/eligibility.py
touch backend/api/spending.py
touch backend/api/assistant.py
touch backend/api/auth.py
touch backend/api/user.py

# Backend Models
touch backend/models/__init__.py
touch backend/models/eligibility/__init__.py
touch backend/models/eligibility/classifier.py
touch backend/models/eligibility/nlp_model.py
touch backend/models/eligibility/vision_model.py
touch backend/models/spending/__init__.py
touch backend/models/spending/predictor.py
touch backend/models/spending/features.py
touch backend/models/assistant/__init__.py
touch backend/models/assistant/chatbot.py
touch backend/models/assistant/recommender.py

# Backend Services
touch backend/services/__init__.py
touch backend/services/eligibility_service.py
touch backend/services/spending_service.py
touch backend/services/assistant_service.py
touch backend/services/user_service.py
touch backend/services/notification_service.py

# Backend Database
touch backend/database/__init__.py
touch backend/database/models.py
touch backend/database/schemas.py
touch backend/database/connection.py
touch backend/database/migrations.py

# Backend Utils
touch backend/utils/__init__.py
touch backend/utils/logger.py
touch backend/utils/validators.py
touch backend/utils/security.py
touch backend/utils/cache.py

# Frontend files
touch frontend/package.json
touch frontend/tsconfig.json
touch frontend/tailwind.config.js
touch frontend/postcss.config.js
touch frontend/.env.example
touch frontend/next.config.js
touch frontend/README.md

# Frontend source
touch frontend/src/App.tsx
touch frontend/src/index.tsx
touch frontend/src/components/Dashboard.tsx
touch frontend/src/components/EligibilityChecker.tsx
touch frontend/src/components/SpendingForecast.tsx
touch frontend/src/components/HealthAssistant.tsx
touch frontend/src/components/Navbar.tsx
touch frontend/src/components/Footer.tsx
touch frontend/src/pages/Home.tsx
touch frontend/src/pages/Dashboard.tsx
touch frontend/src/pages/Profile.tsx
touch frontend/src/pages/Settings.tsx
touch frontend/src/services/api.ts
touch frontend/src/services/auth.ts
touch frontend/src/utils/helpers.ts
touch frontend/src/utils/constants.ts
touch frontend/src/types/index.ts

# ML files
touch ml/requirements.txt
touch ml/notebooks/01_data_exploration.ipynb
touch ml/notebooks/02_eligibility_model.ipynb
touch ml/notebooks/03_spending_prediction.ipynb
touch ml/training/train_eligibility_classifier.py
touch ml/training/train_spending_model.py
touch ml/training/train_assistant.py
touch ml/evaluation/evaluate_eligibility.py
touch ml/evaluation/evaluate_spending.py
touch ml/evaluation/metrics.py
touch ml/data/preprocess.py
touch ml/data/augmentation.py
touch ml/configs/eligibility_config.yaml
touch ml/configs/spending_config.yaml

# Deployment files
touch deployment/docker/Dockerfile.backend
touch deployment/docker/Dockerfile.frontend
touch deployment/docker/docker-compose.yml
touch deployment/docker/docker-compose.prod.yml
touch deployment/kubernetes/backend-deployment.yaml
touch deployment/kubernetes/frontend-deployment.yaml
touch deployment/kubernetes/postgres-deployment.yaml
touch deployment/kubernetes/redis-deployment.yaml
touch deployment/kubernetes/ingress.yaml
touch deployment/kubernetes/configmap.yaml
touch deployment/kubernetes/secrets.yaml
touch deployment/terraform/main.tf
touch deployment/terraform/variables.tf
touch deployment/terraform/outputs.tf

# Test files
touch tests/__init__.py
touch tests/conftest.py
touch tests/unit/test_eligibility.py
touch tests/unit/test_spending.py
touch tests/unit/test_assistant.py
touch tests/integration/test_api.py
touch tests/integration/test_database.py
touch tests/e2e/test_user_flow.py

# Documentation
touch docs/ARCHITECTURE.md
touch docs/API.md
touch docs/DEPLOYMENT.md
touch docs/CONTRIBUTING.md

# Root level files
touch .gitignore
touch .env.example
touch docker-compose.yml
touch Makefile
touch LICENSE
touch CONTRIBUTING.md

echo "📄 Files created"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.*.local

# Database
*.db
*.sqlite3

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*

# Node.js
node_modules/
.npm
.eslintcache

# Frontend build
/frontend/build
/frontend/dist
/frontend/.next
/frontend/out

# Testing
.coverage
.pytest_cache/
htmlcov/
coverage.xml
*.cover

# ML Models
*.h5
*.pkl
*.pth
*.onnx
checkpoints/

# Data
/ml/data/raw/
/ml/data/processed/
*.csv
*.parquet

# Docker
*.tar

# Secrets
secrets/
*.pem
*.key
*.crt

# Terraform
.terraform/
*.tfstate
*.tfstate.backup

# macOS
.DS_Store

# Temporary files
*.tmp
*.temp
EOF

echo "✅ Workspace setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run this script: chmod +x setup_workspace.sh && ./setup_workspace.sh"
echo "2. Let the AI populate all the files with code"
echo "3. Install dependencies: cd backend && pip install -r requirements.txt"
echo "4. Install frontend deps: cd frontend && npm install"
echo ""
echo "🚀 Ready to build the IHC Platform!"