# create env
uv venv --python 3.12 --no-project
 
# activate environment (Windows / Linux)
./.venv/Scripts/activate
source ./.venv/bin/activate

# init isolated project
uv init . --python 3.12 --no-workspace
uv add -r .\requirements.txt

