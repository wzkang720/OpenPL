# How to training and evaluating promtp learning methods

## Environment

[Dassl.ProGrad.pytorch](https://github.com/htyao89/KgCoOp/tree/main/Dassl.ProGrad.pytorch) is the modified toolbox of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). And you can follow the installation of Dassl.ProGrad.pytorch

## Datasets

Follow the instructions in [DATASETS.md](DATASETS.md) to preprocess the datasets.

## Training and testing

For motheds including CoOp, CoCoOp, VPT, MaPLe, PromptSRC and ProDA, you can go to the [6-prompt](./6-prompts) and run the corresponding training and testing script in the [scripts](./6-prompts/scripts) of [6-prompt](./6-prompts). You can also choose to use a specific GPU. All the training scripts are designed to train the algorithm on base classes. Below we provide an example on how to run MaPLe on cuda:1.

```
cd 6-prompts
bash scripts/maple/base2new_train_maple.sh 1
```
and an example on how to test MaPLe on cuda:1

```
cd 6-prompts
bash /prompt-learning/6-prompts/scripts/maple/base2new_test_maple.sh 1
```
For methods including TCP, ProGrad and KgCoOp, You can also run the corresponding script in the [scripts](./textual-prompts/scripts) under the [textual-prompts](./textual-prompts), while choosing the GPU to use. For example,you can train and test TCP.

```
cd textual-prompt
bash scripts/tcp/base2new_train_tcp.sh 0
bash scripts/tcp/base2new_test_tcp.sh 0
```

For RPO, just go to the [RPO](./RPO) and run the scripts like

```
cd RPO
bash scripts/rpo/base2new_train.sh 1
bash scripts/rpo/base2new_test.sh 1
```


## Evaluations
You can use the [cp.py](6-prompts/cp.py) in the folder to extract all log documents from the obtained outputs. Then you can use [res.ipynb](res.ipynb) to evaluate all the results, get evaluation results under various matrics, and generate charts. The results of training and testing is under res/. Say the structure of res/ is

```
output
|–– caltech101/
|   |–– CoOp/
|   |   |–– base/
|   |   |–– new1/
|   |   |   |–– seed1/
|   |   |   |–– seed2/
|   |   |   |–– seed3/
|   |   |   |   |–– log.txt/
|   |   |–– new_ratio1/
|   |   |   |–– seed1/
|   |   |   |–– seed2/
|   |   |   |–– seed3/
|   |   |   |   |–– log.txt/
```