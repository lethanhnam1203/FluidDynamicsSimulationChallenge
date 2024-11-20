# FluidDynamicsSimulationChallenge
Using neural networks to solve a simulation problem in fluid dynamics


## Create a virtual environment

We use python version 3.11.9

After having the right python version, create a new virtual environment (in our case named `thai`) and activate it.
Then install the necessary packages listed in `requirements.txt`

```bash
python -m venv thai
source thai/bin/activate
pip install -r requirements.txt
```

## Run models

Execute the bash script `run_model.sh` at root.

```bash
./run_models.sh
```

If there is any permission error related to this bash script, you may need to grant execute permissiosn via:

```bash
chmod +x run_models.sh
```
