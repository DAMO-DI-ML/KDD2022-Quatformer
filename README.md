# Quatformer (KDD 2022 paper)

* Weiqi Chen, Wenwei Wang, Bingqing Peng, Qingsong Wen, Tian Zhou, Liang Sun, "Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting" in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2022), 2022. [[paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539234)]

Quaternion Transformer (Quatformer) introduce quternion to model complicated periodical patterns (i.e., muliple periods, variable periods, and phase shift) in time series which also has a linear complexity with decoupling attention. Our empirical studies with six benchmark datasets verify its effectiveness. 

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Install other dependencies by:
```shell
pip install -r requirements.txt
```
3. Download data. You can obtain all the six benchmarks from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].
4. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/ETT_script/Quatformer.sh
bash ./scripts/ECL_script/Quatformer.sh
bash ./scripts/Exchange_script/Quatformer.sh
bash ./scripts/Traffic_script/Quatformer.sh
bash ./scripts/Weather_script/Quatformer.sh
bash ./scripts/ILI_script/Quatformer.sh
```


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{chen2022quatformer,
  title={Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting},
  author={Chen, Weiqi and Wang, Wenwei and Peng, Bingqing and Wen, Qingsong and Zhou, Tian and Sun, Liang},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={146--156},
  year={2022}
}
```



## Contact

If you have any question or want to use the code, please contact jarvus.cwq@alibaba-inc.com.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data
