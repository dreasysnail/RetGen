
import os,sys
import logging
from functools import partial

from demo_utils import download_model_folder
import argparse
import subprocess as sp

PYTHON_EXE = 'python'
MODEL_FOLDER = './models'
DATA_FOLDER = './data'

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='dummy',
                    help='choose from dummy, small and full')
dargs = parser.parse_args()

assert dargs.data == 'dummy' or dargs.data == 'small' or dargs.data == 'full' , \
    'The specified data option is not support!'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)

if os.path.exists(MODEL_FOLDER):
    print('Found existing ./models folder, skip creating a new one!')
    os.makedirs(MODEL_FOLDER, exist_ok=True)
else:
    os.makedirs(MODEL_FOLDER)

logger.info('Downloading models...')
download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)

target_folder = download_model(model_size='small', dataset='multiref', from_scratch=False)
logger.info('Done!\n')

logger.info('Downloading and Extracting Data...')
if dargs.data == 'dummy':
    cmd = 'bash prepare4db.sh'
    ret = sp.run(cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT, cwd=DATA_FOLDER)
elif dargs.data == 'small':
    myCmd = os.popen('cd reddit_extractor; make -j 8; cd ..').read()
    cmd = 'gzip -d ./train.tsv.gz'
    ret = sp.run(cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT, cwd=DATA_FOLDER)
elif dargs.data == 'full':
    myCmd = os.popen('cd reddit_extractor; SIZE=full make -j 8; cd ..').read()
    cmd = 'gzip -d ./train.tsv.gz'
    ret = sp.run(cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT, cwd=DATA_FOLDER)
data_path = os.path.join(DATA_FOLDER, 'train.tsv')

logger.info('Preparing Data...')

MAX_LEN = 128
cmd = ['prepro.py', '--corpus', data_path, '--max_seq_len', f'{MAX_LEN}']
cmd = ' '.join(cmd) 
print(cmd)
ret = sp.run([PYTHON_EXE] + cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT)

data_db = f'{data_path[:-4]}.{MAX_LEN}len.db'

logger.info('Done!\n')

logger.info('Generating training CMD!')
logger.info('If there is any problem, please copy (modify) and run command below')
logger.info('
train_cmd = 'LSP_train.py'
args = [
    '--model_name_or_path', target_folder,
    '--init_checkpoint', os.path.join(target_folder, 'pytorch_model.bin'),
    '--train_input_file', data_db ,  
    '--eval_input_file', './data/dummy_data.tsv',   
    '--output_dir', os.path.join(MODEL_FOLDER, 'output_model'),
    '--seed', '42',
    '--max_seq_length', '128',
    '--train_batch_size', '512',
    '--gradient_accumulation_steps', '8',
    '--eval_batch_size', '64',
    '--learning_rate', '1e-5',
    '--num_optim_steps', '10000',
    '--valid_step', '5000',
    '--warmup_steps', '4000',
    '--normalize_data', 'true',
    '--fp16', 'true',
    '--lr_schedule', 'noam',
    '--loss_scale', '0.0',
    '--no_token_id', 'true',
    '--pbar', 'true'
]

arg = ' '.join(args)
train_cmd = train_cmd + ' ' + arg
print(PYTHON_EXE + ' ' +train_cmd)
logger.info('
with open('./output.log', 'wb') as f: 
    process = sp.Popen([PYTHON_EXE] + train_cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT)
    for line in iter(process.stdout.readline, b''): 
        sys.stdout.write(line.decode(sys.stdout.encoding)) 
        f.write(line)
logger.info('Done!\n')
