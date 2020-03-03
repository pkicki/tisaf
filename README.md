# Neural Path Planning

This repository contains code associated with the paper "[A Self-Supervised Learning Approach to Rapid Path Planning for Car-Like Vehicles Maneuvering in Urban Environment](https://arxiv.org/abs/2003.00946)".

```
An efficient path planner for autonomous car-like vehicles should handle the strong kinematic constraints,
particularly in confined spaces commonly encountered while maneuvering in city traffic,
and should enable rapid planning, as the city traffic scenarios are highly dynamic.
State-of-the-art planning algorithms handle such difficult cases at high computational cost,
often yielding non-deterministic results.
However, feasible local paths can be quickly generated leveraging the past planning experience
gained in the same or similar environment. While learning through supervised training is problematic
for real traffic scenarios, we introduce in this paper a novel neural network-based method for path planning,
which employs a gradient-based self-supervised learning algorithm to predict feasible paths.
This approach strongly exploits the experience gained in the past and rapidly yields feasible maneuver plans
for car-like vehicles with limited steering-angle.
The effectiveness of such an approach has been confirmed by computational experiments.
```

Name of the repository `tisaf` is a shortcut for the claim from the paper: "There Is Such A Function that for any input task, for which there exist valid path solving that task, returns a valid path". 

## Dependencies

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

