# LTSF_Linear

Competition URL:
https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge


## How to run and make prediction

All the codes are available to run in Kaggle workspace

```bash
!git clone https://github.com/HoangVo-Prog/LTSF_Linear.git
```

### Rolling Elastic Net 
```bash
!python LTSF_Linear/src/rolling_elastic_net/main.py
```


### OLS Elastic Net

```bash
!python LTSF_Linear/src/ols_elastic_net/main.py 
```

### Pipeline Direct

```bash 
!python LTST_Linear/src/pipeline/main_pipeline_direct.py
```

### PatchTST (Our best result: 15.0147 mse)

```bash
# Make sure to turn on GPU
!python LTST_Linear/src/patchtst/main.py
```