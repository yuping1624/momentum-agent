# Code Quality Guide

## Flake8 Setup

This project uses **flake8** for code quality checking.

### Installation

Flake8 is already included in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

Or install flake8 directly:

```bash
pip install flake8
```

### Configuration

The project includes a `.flake8` configuration file with the following settings:

- **Max line length**: 120 characters
- **Ignored errors**: E203, E501, W503 (conflicts with formatters)
- **Max complexity**: 10
- **Excluded directories**: venv, logs, data, build, dist, etc.

### Usage

#### Check all Python files:
```bash
python3 -m flake8 .
```

#### Check specific files:
```bash
python3 -m flake8 app.py brain.py test.py
```

#### Show statistics:
```bash
python3 -m flake8 . --statistics
```

#### Only show errors (hide warnings):
```bash
python3 -m flake8 . --select=E
```

#### Auto-fix format issues (using autopep8):
```bash
# Install autopep8
pip install autopep8

# Auto-fix format issues
autopep8 --in-place --aggressive --aggressive app.py brain.py test.py
```

### Common Flake8 Error Codes

- **E302**: Expected 2 blank lines before function/class definition
- **E501**: Line too long
- **F401**: Module imported but unused
- **W293**: Blank line contains whitespace
- **W291**: Trailing whitespace
- **C901**: Function is too complex

### Pre-commit Hook (Optional)

You can set up a pre-commit hook to automatically check code before commits:

```bash
# Create .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
python3 -m flake8 app.py brain.py test.py check_models.py
EOF

chmod +x .git/hooks/pre-commit
```

