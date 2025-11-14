# Contributing to IHC Platform

Thank you for your interest in contributing to the Intelligent Health Checkout Platform! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker & Docker Compose
- Git

### Setup Development Environment

1. **Fork and clone the repository:**
```bash
git clone https://github.com/yourusername/IHC-NationalHealthAI.git
cd IHC-NationalHealthAI
```

2. **Run setup:**
```bash
make setup
```

3. **Start development servers:**
```bash
make dev
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch Naming Convention:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
make test

# Run linters
make lint

# Format code
make format
```

### 4. Commit Your Changes

**Commit Message Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Example:**
```bash
git commit -m "feat(eligibility): add support for dental procedures

Implemented eligibility checking for common dental procedures
including cleanings, fillings, and orthodontics.

Closes #123"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Pull Request Guidelines

### PR Title Format

```
<type>: <description>
```

Example: `feat: Add spending forecast export feature`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally

## Screenshots (if applicable)

## Related Issues
Closes #<issue_number>
```

### Review Process

1. **Automated Checks:** CI/CD pipeline runs tests and linters
2. **Code Review:** At least one maintainer review required
3. **Testing:** Verify functionality in staging environment
4. **Approval:** Maintainer approval required for merge

## Coding Standards

### Python (Backend)

- Follow PEP 8 style guide
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for all functions/classes

**Example:**
```python
def predict_spending(
    user_features: Dict[str, any],
    historical_data: Optional[pd.DataFrame] = None
) -> Dict[str, any]:
    """Predict annual healthcare spending.
    
    Args:
        user_features: User demographic and health information
        historical_data: Historical spending data
        
    Returns:
        Prediction results with confidence intervals
    """
    # Implementation
    pass
```

### TypeScript (Frontend)

- Use TypeScript for all new code
- Follow Airbnb style guide
- Use functional components with hooks
- Proper prop typing

**Example:**
```typescript
interface SpendingForecastProps {
  userId: string;
  onUpdate?: (data: ForecastData) => void;
}

export const SpendingForecast: React.FC<SpendingForecastProps> = ({
  userId,
  onUpdate
}) => {
  // Implementation
};
```

### Testing

**Backend Tests:**
```python
import pytest

def test_eligibility_classifier():
    """Test eligibility classifier accuracy."""
    classifier = EligibilityClassifier()
    result = classifier.predict("Prescription eyeglasses")
    
    assert result["is_eligible"] is True
    assert result["confidence"] > 0.9
```

**Frontend Tests:**
```typescript
import { render, screen } from '@testing-library/react';

describe('Dashboard', () => {
  it('renders user balance', () => {
    render(<Dashboard userId="test-123" />);
    expect(screen.getByText(/balance/i)).toBeInTheDocument();
  });
});
```

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Include type hints
- Explain complex algorithms
- Add inline comments for non-obvious code

### API Documentation

- Document all endpoints in OpenAPI format
- Include request/response examples
- Document error codes
- Keep API docs up to date

### README Updates

- Update README for new features
- Add usage examples
- Update installation instructions if needed

## Security

### Reporting Security Issues

**DO NOT** create public GitHub issues for security vulnerabilities.

Instead, email: security@ihc-platform.com

### Security Best Practices

- Never commit secrets or API keys
- Use environment variables for sensitive data
- Validate all user inputs
- Follow HIPAA compliance guidelines
- Encrypt PHI data

## ML Model Contributions

### Model Improvements

1. **Benchmark:** Test against current model performance
2. **Documentation:** Document model architecture and training
3. **Reproducibility:** Include training scripts and configs
4. **Evaluation:** Provide comprehensive evaluation metrics

### Training Data

- Use publicly available datasets (MEPS, IRS data)
- Document data preprocessing steps
- Include data validation checks
- Respect privacy and compliance requirements

## Community

### Communication Channels

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and ideas
- **Slack:** Real-time chat (link in README)
- **Email:** contact@ihc-platform.com

### Getting Help

- Check existing issues and documentation
- Ask in GitHub Discussions
- Join our Slack community
- Attend monthly contributor meetings

## Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Eligible for contributor swag

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to IHC Platform! Your efforts help improve healthcare accessibility for millions of Americans. 🎉