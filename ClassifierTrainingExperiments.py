import os

img_in_epoch = 100000

for unit_type in range(3):
    dirname = f"classifiers_architecture{unit_type}"
    os.makedirs(dirname)
    for dataset in ["MNIST", "CelebA128Gender", "LSUN128"]:
        def run(load_predix: Optional[str], save_prefix: str, more: str):
            save_filename = f"{dirname}/{save_predix}_{dataset}.bin"
            load_filename = f"{dirname}/{load_predix}_{dataset}.bin"
            save_cmd = f'--save_filename \"{load_filename}"'
            load_cmd = f'--load_filename \"{load_filename}"' if load_predix else ""
            os.system(f'ipython ClassifierTraining.py -- --dataset {dataset} --command train '
                      f'{load_cmd} {same_cmd} --img_in_epoch {img_in_epoch} --unit_type {unit_type} {more}')
        
        run(None,           "oneepoch",     "--no_epochs 1")
        run("oneepoch",     "plain",        "--no_epochs 7 --start_lr 0.0003")
        run(None,           "conventional", "")
        run("plain",        "robust",       "")
        run("conventional", "both",         "")