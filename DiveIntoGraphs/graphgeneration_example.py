import json
from dig.ggraph.dataset import ZINC250k
from torch_geometric.loader import DenseDataLoader
conf = json.load(open('config/rand_gen_zinc250k_config_dict.json'))
dataset = ZINC250k(one_shot=False, use_aug=True)
loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)

from dig.ggraph.method import GraphDF
runner = GraphDF()
lr = 0.001
wd = 0
max_epochs = 10
save_interval = 1
save_dir = 'rand_gen_zinc250k'
runner.train_rand_gen(loader=loader, lr=lr, wd=wd, max_epochs=max_epochs,
    model_conf_dict=conf['model'], save_interval=save_interval, save_dir=save_dir)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
ckpt_path = 'rand_gen_zinc250k/rand_gen_ckpt_10.pth'
n_mols = 100
mols, _ = runner.run_rand_gen(model_conf_dict=conf['model'], 
    n_mols=n_mols, atomic_num_list=conf['atom_list'])

from dig.ggraph.evaluation import RandGenEvaluator
evaluator = RandGenEvaluator()
input_dict = {'mols': mols}
print('Evaluating...')
evaluator.eval(input_dict)