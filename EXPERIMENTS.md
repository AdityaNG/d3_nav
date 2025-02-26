# D3Nav v2 Experiemtns

We run the following exeriments:


| Model | L2 1s | L2 2s | L2 3s | Train Loss | Val Loss |
|-------|--------|--------|--------|------------|-----------|
| ResNet (Pure) | 1.39 | 3.0 | 4.5 | 4.4 | 2.5 |
| ResNet-Traj (Frozen) | 1.13 | 2.0 | 2.8 | 1.0 | 1.0 |
| ResNet-Traj-ft (Unfrozen) | 1.08 | 2.0 | 3.0 | 1.0 | 1.1 |
| D3Nav-3L | 1.14 | 1.4 | 2.0 | 0.8 | 0.8 |
| D3Nav-3L-ft | 0.76 | 1.4 | 2.0 | 0.3 | 0.8 |
| D3Nav-3L-ft-CA | 0.66 | 1.2 | 1.8 | 0.2 | 0.7 |
| D3Nav-3L-ft-CA-D0.2 | 1.00849 | 1.7547 | 2.5741 | 0.3725 | 0.96 |
| D3Nav-3L-ft-CA-T2 | 0.71 | 1.25 | 1.867 | 0.26 | 0.71 |
| D3Nav-3L-ft-CA-T1 | 1.417 | 2.37 | 3.35 | 0.1462 | 1.232 |

Notes:
- First we pretrain our `TrajectoryEncoder` and `TrajectoryDecoder` on the task of trajectory reconstruction. We use these latents as an interface for our models to predict trajectory.
- We start of by training baseline models using the ResNet family 
    - The baseline runs get to a minimum L2 (1s) of ~1.08
    - Our baseline runs showed us that using the pretrained trajectory decoder helps, unfreezing the decoder helps even further
- D3Nav-3L: We fine tune the world model GPT backbone for the task of driving by unfreezing the last 3 layers and taking the BOS token and feeding that into the trajectory decoder. This put a heavy load on the single token and we reached an L2 (1s) of 1.14
- D3Nav-3L-ft: Taking our learnings from the baseline experiment, we unfreeze the trajectory decoder, this improves the performance drastically as the backbone is no longer bottlenecked by its ability to interface with the frozen trajectory decoder. This brings us to L2 (1s) pf 0.76
- D3Nav-3L-ft-CA: in order to reduce the load on the single token predicting the trajectory, we use the entire last layer of the transformer and we apply `ChunkedAttention`. This further improves our performance to L2 (1s) of 0.66
- D3Nav-3L-ft-CA-D0.2: then we apply dropout to the inputs to help the model generalize further, but this dropout might have been too agressive. We have put this on pause for now (future work)
- Causal Confusion: the temporal models of 8 frames seem to be cheating by interpolating the past ego motion to infer the desired trajectory. We now try a 2 frame and single frame model to verify this hypothesis:
    - D3Nav-3L-ft-CA-T2: Gets a low L2 of 0.71, indicating that all the 8 frames are not needed for causal confusion in such a model. The behaviour still seems to indicate that the model is cheating
    - D3Nav-3L-ft-CA-T1: When we remove all temporal context, the model finally stops cheating. However, the L2 loss shoots up to 1.4. We should look into addressing causal confusion by training our planner to work regardless of the number of frames we provide to the transformer as input.

Key observations:
1. D3Nav outperforms the ResNet baseline
2. Adding the trajectory decoder (ResNet-Traj) significantly improves performance
3. The D3Nav variants show progressive improvements
4. D3Nav-3L-ft-CA-D0.2 dropout rate was too agressive
5. Input dropout doesn't work, we will have to alter the training pipeline to simply not feed in temporal context as input.

## Baseline ResNet-34

ResNet: Pure ResNet regressing 6x2 trajectory

![ResNet](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/resnet_pure.png)

ResNet-Traj: ResNet with frozen Trajectory Decoder
![ResNet-Traj](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/resnet_traj_decoder_frozen.png)

ResNet-Traj-ft: ResNet with unfrozen Trajectory Decoder fine tuned
![ResNet-Traj](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/resnet_traj_decoder_unfrozen.png)

# D3Nav

[D3Nav-3L](https://wandb.ai/adityang/D3Nav-NuScenes/runs/hcnqal2v?nw=nwuseradityang): Unfrozen 3 Layers, single token, frozen Trajectory Decoder
![D3Nav-3L](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/D3Nav-3L.png)

[D3Nav-3L-ft](https://wandb.ai/adityang/D3Nav-NuScenes/runs/czr85wgs?nw=nwuseradityang): Unfrozen 3 Layers, single token, unfrozen Trajectory Decoder
![D3Nav-3L-ft](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/D3Nav-3L-ft.png)

[D3Nav-3L-ft-CA](https://wandb.ai/adityang/D3Nav-NuScenes/runs/fv05dza3?nw=nwuseradityang): Unfrozen 3 Layers, unfrozen Trajectory Decoder, chunked attention
![D3Nav-3L-ft-CA](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/D3Nav-3L-ft-CA.png)

[D3Nav-3L-ft-CA-D0.2](https://wandb.ai/adityang/D3Nav-NuScenes/runs/fv05dza3?nw=nwuseradityang): Unfrozen 3 Layers, unfrozen Trajectory Decoder, chunked attention, with frame level dropout of 20%
![D3Nav-3L-ft-CA-D0.2](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/D3Nav-3L-ft-CA-D0.2.png)

[D3Nav-3L-ft-CA-T2](https://wandb.ai/adityang/D3Nav-NuScenes/runs/w1e3j3yn): Unfrozen 3 Layers, unfrozen Trajectory Decoder, chunked attention, with frame level dropout of 20%, 2 temporal frames, trying to address the causal confusion, not much luck, even only 2 frames causes it to cheat
![D3Nav-3L-ft-CA-T2](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/D3Nav-3L-ft-CA-T2.png)

[D3Nav-3L-ft-CA-T1](https://wandb.ai/adityang/D3Nav-NuScenes/runs/w1e3j3yn): Unfrozen 3 Layers, unfrozen Trajectory Decoder, chunked attention, with frame level dropout of 20%, 1 frames only, addressed the causal confusion
![D3Nav-3L-ft-CA-T1](https://github.com/AdityaNG/d3_nav/raw/main/media/runs/D3Nav-3L-ft-CA-T1.png)
