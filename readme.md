# OpenPL
This is a benchmark for prompt learning in VLM based on Dynamic Classes Changes, Dynamic Distribution Shifts and Dynamic Co-evolution of Distribution and Class Variation.

## Prompt learning methods
Including [CoOp](https://github.com/KaiyangZhou/CoOp), [CoCoOp](https://github.com/KaiyangZhou/CoOp), VPT, [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [PromptSRC](https://github.com/muzairkhattak/PromptSRC), [ProDA](https://github.com/bbbdylan/proda), [TCP](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning), [ProGrad](https://github.com/BeierZhu/Prompt-align), [KgCoOp](https://github.com/htyao89/KgCoOp) and [RPO](https://github.com/mlvlab/RPO).
# How to training and evaluating prompt learning methods

## Environment

* Dassl.pytorch is used for most of prompt learning methods and we modified the data read file in it for our experiments. And you can follow the installation of Dassl.pytorch

* [Dassl.ProGrad.pytorch](https://github.com/htyao89/KgCoOp/tree/main/Dassl.ProGrad.pytorch) is the modified toolbox of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), which is only used for RroGrad method . And the installation is the same to Dassl.pytorch.

## Datasets

Follow the instructions in [DATASETS.md](DATASETS.md) to preprocess the datasets.

## Training and testing

For motheds including CoOp, CoCoOp, VPT, MaPLe, PromptSRC and ProDA, you can go to the [6-prompt](./6-prompts) and run the corresponding training and testing script in the [scripts](./6-prompts/scripts) of [6-prompt](./6-prompts). You can also choose to use a specific GPU. All the training scripts are designed to train the algorithm on base classes. Below we provide an example on how to run MaPLe on cuda:1.

```bash
cd 6-prompts
bash scripts/maple/base2new_train_maple.sh 1
```
and an example on how to test MaPLe on cuda:1

```bash
cd 6-prompts
bash scripts/maple/base2new_test_maple.sh 1
```
for Dynamic Distribution Shifts and Dynamic Co-evolution of Distribution and Class Variation, you can use these scripts to reproduce.

```bash
cd 6-prompt
bash scripts/maple/batch_xd_train.sh
bash scripts/maple/batch_xd_train.sh
```

 Note:

We use the following hyperparameters to control the currently running task in train.py. The default task is Dynamic Classes Changes; mix_distribution represents the Dynamic Distribution Shifts; cross_dataset represents Dynamic Co-evolution of Distribution and Class Variation.

```bash
   parser.add_argument(
        "--mix_distribution", action="store_true", help="do call mix distribution"
    )
    parser.add_argument(
        "--cross_dataset", action="store_true", help="do call cross dataset"
    )
```



For methods including TCP, ProGrad and KgCoOp, You can also run the corresponding script in the [scripts](./textual-prompts/scripts) under the [textual-prompts](./textual-prompts), while choosing the GPU to use. For example,you can train and test TCP.

```bash
cd textual-prompt
bash scripts/tcp/base2new_train_tcp.sh 0 
bash scripts/tcp/base2new_test_tcp.sh 0
```

For RPO, just go to the [RPO](./RPO) and run the scripts like

```bash
cd RPO
bash scripts/rpo/base2new_train.sh 1
bash scripts/rpo/base2new_test.sh 1
```

Other experiments are similar to these above.

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
