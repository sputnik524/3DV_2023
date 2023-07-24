## Run MA-RLSAC with

### For standard MA-RLSAC training
```
python3 src/main.py --config=rlsac_coma_ns --env-config=rlsac with env_args.time_limit=25 env_args.key='SACEnv-v1'
```

### For shared parameter MA-RLSAC training

```
python3 src/main.py --config=rlsac_coma_halfshared --env-config=rlsac with env_args.time_limit=25 env_args.key='SACEnv-v1'
```

### For standard MA-RLSAC testing

1. Change two config <code>.yaml</code>files <br>
    1.1 <code>config/env/rlsac.yaml</code> <br>
        change the <code>test_nepisode</code> to 9900 in this config file
        
    1.2 <code>config/algs/rlsac_coma_ns.yaml</code> <br>
        uncomment line 45,46 and change the checkpoint loading path to the trained model, which is stored in <code>../results/models</code>. If doing shared parameter MA-RLSAC, <code>config/algs/rlsac_coma_halfshared.yaml</code> <br>
        uncomment line 45,46 and change the checkpoint loading path to the trained model, which is stored in <code>../results/models</code>

2. Run the same command as the training one




# Acknowledgement

* Credits to Extended Python MARL framework - [EPyMARL](https://agents.inf.ed.ac.uk/blog/epymarl/) 
   ```
   @inproceedings{papoudakis2021benchmarking,
      title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
      author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
      booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
      year={2021},
      url = {http://arxiv.org/abs/2006.07869},
      openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
      code = {https://github.com/uoe-agents/epymarl},
   }
   ```


    
