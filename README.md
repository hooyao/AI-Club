# AI-Club

## 怎样选择合适的开发平台

![choose_ai_plat]

[choose_ai_plat]: https://github.com/BaiwangTradeshift/AI-Club/raw/master/doc/images/How%20to%20Choose%20a%20AI%20Dev%20Platform.png "How to Choose a AI Dev Platform"

## 怎样搭建开发环境
### MAC
1. 安装miniconda
2.
```sh
conda create -n aiclub python=3.6
source activate aiclub
conda install pandas scikit-learn scikit-image scipy matplotlib sympy jupyter nb_conda -y
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl
pip install tflearn
```
或者
```sh
conda env create -f aiclub_mac.yml
source activate aiclub
```
3. jupyter notebook --ip='*' --NotebookApp.token= --port=8888
4. OK
