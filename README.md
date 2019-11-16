# Neural Planning

This repository contains work on the path planning with the use of neural networks.

## Dependencies

* dl_work repository (PPI Lab),
* Tensorflow 1.13+ (Eager Execution)

## Before run

Create "experiments/working_dir" directory. 

## How to run

```bash
cd experiments
python planner.py --config-file ./config_files/XXX.conf
```

## Training scheme
1. Uncomment loss in models/planner.py for pretraining
2. Run training with the config_file you like
3. Find the trained model in working_dir/<<<out-name>>>/checkpoints
4. In experiments/planner.py add restore action with the path to chosen model
5. Uncomment loss in models/planner.py for training
6. Run training with the config_file you like

