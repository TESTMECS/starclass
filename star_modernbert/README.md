# Usage

## Install Packages:
### [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
#### Linux & Mac:
`curl -LsSf https://astral.sh/uv/install.sh | sh`
`wget -qO- https://astral.sh/uv/install.sh | sh`
#### Windows:
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

# Sync Packages: 
`cd star_modernbert`
`uv init`
`uv sync --project pyproject.toml`

# Install Dependencies:
## Install transformers:
`uv pip install git+https://github.com/huggingface/transformers`
## Install flash-attn:
`uv pip install flash-attn`

# To Activate venv:
    `source .venv/bin/activate`

# To Deactivate venv:
    `deactivate`
    `.venv/bin/deactivate`

## To Test:
    `uv run pipeline.py <sentence>`
with venv activated use:
    `python pipeline.py <sentence>`

Can either leave it blank for a pre-defined sentence or input your own sentence.
If you want to use a list of sentences check the comment for an example. 

    

