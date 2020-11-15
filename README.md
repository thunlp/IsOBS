# IsOBS

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Background

本项目应用论文prototypical-networks-for-few-shot-learning中所描述的原型网络来处理甲骨文的分类问题。

## Install

下载代码后即可直接运行

## Usage

运行`python3 ./train.py`使用默认参数进行训练，训练出来的模型默认保存在`model_save`目录下。下列是一些重要参数。

- `--oracle [300/600/1600]` 选择使用规模多大的数据集

- `--epochs` 设置训练的epoch数

更多训练相关参数请使用`python3 ./train.py --help`查看。

## Contributing

PRs not accepted.

## License

MIT © Richard McRichface
