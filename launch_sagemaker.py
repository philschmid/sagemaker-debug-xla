from datasets import Dataset
from transformers import AutoTokenizer
import os
import pandas as pd
import sagemaker
import botocore

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_PROFILE"] = "hf-sm"

tokenizer_name = 'roberta-base'
os.system('aws s3 sync "s3://sagemaker-sample-files/datasets/text/SST2/" "./"')

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# load dataset
test_df = pd.read_csv('sst2.test', sep='delimiter', header=None, engine='python', names=['line'])
train_df = pd.read_csv('sst2.train', sep='delimiter', header=None, engine='python', names=['line'])
test_df[['label', 'text']] = test_df['line'].str.split(' ', 1, expand=True)
train_df[['label', 'text']] = train_df['line'].str.split(' ', 1, expand=True)
test_df.drop('line', axis=1, inplace=True)
train_df.drop('line', axis=1, inplace=True)
test_df['label'] = pd.to_numeric(test_df['label'], downcast='integer')
train_df['label'] = pd.to_numeric(train_df['label'], downcast='integer')
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# tokenize dataset
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# set format for pytorch
train_dataset =  train_dataset.rename_column("label", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

save_path = "/opt/ml/model"

optimized_estimator = PyTorch(
        entry_point="train.py", #see below
        source_dir="./scripts",
        instance_type="ml.g4dn.2xlarge",
        instance_count=1,
        role="admin",
        py_version="py38",
        framework_version="1.12.0",
        volume_size=200,
        hyperparameters={
            'epochs': 3,
            'max_steps': 5.
            'train_batch_size': 24,
            'model_name': 'roberta-base',
            'optim': 'adamw_torch_xla',
        } ,
        disable_profiler=True,
        debugger_hook_config=False,
        max_retry_attempts=3,
        compiler_config=TrainingCompilerConfig(debug=True),
        environment={
            "XLA_FLAGS": f"--xla_dump_to={save_path} --xla_dump_hlo_as_text",
            "XLA_SAVE_TENSORS_FILE": f"{save_path}/XLA_SAVE_TENSORS_FILE.hlo",
            "XLA_METRICS_FILE": f"{save_path}/XLA_METRICS_FILE.txt",
            "XLA_SAVE_HLO_FILE": f"{save_path}/XLA_SAVE_HLO_FILE.hlo",
        },
        distribution={'pytorchxla': {'enabled': True}},
    )

    # starting the train job with our uploaded datasets as input
optimized_estimator.fit({"train": training_input_path, "test": test_input_path}, wait=True)