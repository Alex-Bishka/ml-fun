python -m venv env
pip install ipykernel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" trackio

python -m ipykernel install --user --name="env" --display-name="gpt"