set -a
if [ -f .env ]; then
    . .env
else
    echo "hint: no .env file found — create one with your API keys, e.g.:"
    echo "  echo 'WANDB_API_KEY=your_key_here' > .env"
fi
PYTHONWARNINGS="ignore::UserWarning:pydantic"
set +a