'test3 폴더'가 실제로 사용한 소스코드입니다.
1. File
config : training scenario parameter(체력, 관광지 등)를 포함한 json 파일
logs : tensorflow로 훈련 과정, 평가 과정 기록(tensorboard로 확인)
makeathon : 발표ppt
models : grid_image_best(training 중, moving average = 50 기준, 완주성공이 가장 많이 되었을 때의 model weight 저장), grid_image_unfinished(training step 완주 시 model weight 저장), eval할 때는 grid_image_best 사용
res : png파일로 만든 RGB 형태의 소이산 grid_image

2. ipynb
Agent : keras를 이용해 model layer 생성, model 구조, 학습 파라미터 등 설정

BaseDisplay : RGB 정보로 grid image 생성, 에피소드 시작~종료 지점을 포함한 경로 도식

BaseGrid : 체력정보포함(json file에도 동시에 포함됨), RGB정보에 대해 맵 정보 정의

BaseState : RGB png file 경로 정의, RGB정보를 포함한 기본적인 state 상태 정의

DeviceManager : device = tour, tour의 색깔, 최초 데이터 정의(IoTDevice 참고)

Display : 한 에피소드 전체 정보(시작, 도착지점, grid_world, tour points, agent 이동 상태) 도식, Replaymemory에 저장된 1개의 batch(state, action, reward, next state 등)를 가져와서 input data로 활용

Environment : training 정의(filling replay memory -> train_episode), training 과정 중에 test_episode 있는 이유 : 기본적인 training setting이 매 에피소드(or epoch)마다 다양한 시나리오 파라미터를 랜덤 선택해서 진행되기 때문에, 
unusally easy conditions 발생 가능(e.g. 3개의 에피소드가 모두 agent가 최대 체력을 선택해서 진행)하기 때문에 multiple evalutions의 평균으로 agent의 학습 진도를 나타낼 수 있음

Grid : tour_points의 좌표 정의(json file 동일), tour의 유형(자연, 인공, 모두) 정의

GridActions : agent의 행동 경우 설정

GridPhysics : agent의 물리적인 step 정의(행동에 따른 state 변화 : 출입금지지역, 도착지점, 힘든지역, 체력감소)

GridRewards : agent를 원하는 방향으로 훈련시키기 위한 reward structure

IoTDevice : tour 별로 최초 데이터 5(DeviceManager 참고)가 있는데, agent가 해당 좌표 이동 시, 데이터 5를 획득, 이 때 각 tour는 data = 0일 때만 positive reward 부여
(data를 이용한 reward 설정 이유 : 만약 이렇게 하지 않으면, agent는 tour 좌표 이동 시 매 step마다 계속 positive reward를 받기 때문에 어느 tour position으로 이동하든 그곳에서부터 이동하지 않을 가능성이 매우 큼)

main_mc : Evaluation using Monte Carlo analysis

MainEnvironment : 현재까지 정의한 모든 모듈을 임포트해서 최종적인 step 등을 정의

Map : RGB의 정보를 각각 이동불가지역, 힘든지역, 시작 및 도착지점으로 분리 정의

ModelStats : log file 저장, Keras의 callback를 이용해 tensorboard에 정보를 연동하기 위한 함수들 정의

Physics : register function (training or evalution을 진행하면서 관찰하고 싶은 정보들에 대한 meric 정의), tour_step(tour 방문 정보를 State.ipynb와 연동)

ReplayMemory : replaymemory에 state, action, reward를 어떻게 저장할지 정의(DDQN이 off-policy이기에 필수적)

Rewards : GridRewards의 reward를 계속 해서 합쳐줌

State : agent의 모든 state 정보를 저장, 다른 모듈과 활발한 연동

StateUtils :  # trianing parameters를 굉장히 많이 줄여줌(우리 환경에서는 3백만->1백만으로 1/3줄어듬), (agent는 fully observable environment가 아니라 partially observabl environment로 현재 위치에 대해 정보가 제한됨 
=> 여러 논문에서 performance의 저하는 없고, 감소된 # training parameters로 훈련 시간을 크게 줄여들어 훈련이 가능하게 함)

Trainer : training 환경 설정

Utils : read_config(json file 불러옴)




## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
* [Resources](#resources)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the double deep Q-learning (DDQN) approach to control multiple UAVs on a data harvesting from IoT sensors mission, including dual global-local map processing. The corresponding paper ["Multi-UAV Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning"](https://arxiv.org/abs/2010.12461) is available on arXiv.

An earlier single-UAV conference version ["UAV Path Planning for Wireless Data Harvesting: A Deep Reinforcement Learning Approach"](https://arxiv.org/abs/2007.00544) is also available on arXiv and was presented at IEEE Globecom 2020.

For questions, please contact [Harald Bayerlein](https://hbay.gitlab.io) via email harald.bayerlein@eurecom.fr. Please also note that due to github's new naming convention, the 'master' branch is now called 'main' branch.


## Requirements

```
python==3.7 or newer
numpy==1.18.5 or newer
keras==2.4.3 or newer
tensorflow==2.3.0 or newer
matplotlib==3.3.0 or newer
scikit-image==0.16.2 or newer
tqdm==4.45.0 or newer
```


## How to use

Train a new DDQN model with the parameters of your choice in the specified config file, e.g. with the standard config for the 'manhattan32' map:

```
python main.py --gpu --config config/manhattan32.json --id manhattan32

--gpu                       Activates GPU acceleration for DDQN training
--config                    Path to config file in json format
--id                        Overrides standard name for logfiles and model
--generate_config           Enable only to write default config from default values in the code
```

For keeping track of the training, use TensorBoard. Various performance and training metrics, as well as intermittent test plots of trajectories, are recorded in log files and automatically saved in the 'logs' directory. On the command line, run:

```
tensorboard --logdir logs
```

Evaluate a model (saved during training in the 'models' directory) through Monte Carlo analysis over the random parameter space for the performance indicators 'Successful Landing', 'Collection Ratio', 'Collection Ratio and Landed' as defined in the paper (plus 'Boundary Counter' counting safety controller activations), e.g. for 1000 Monte Carlo iterations:

```
python main_mc.py --weights models/manhattan32_best --config config/manhattan32.json --id manhattan32_mc --samples 1000

--weights                   Path to weights of trained model
--config                    Path to config file in json format
--id                        Name for exported files
--samples                   Number of Monte Carlo  over random scenario parameters
--seed                      Seed for repeatability
--show                      Pass '--show True' for individual plots of scenarios and allow plot saving
--num_agents                Overrides number of agents range, e.g. 12 for random range of [1,2] agents, or 11 for single agent
```

With the most recent code update, the config options and associated code to train agents with either 'scalar' input (that means no map, but only concatenated numerical values as state information as described in [1]) or 'blind' (only ego UAV position and remaining flying time as state input) were added. These are only relevant for the comparison in paper [1] and can be savely ignored when you are only interested in the map-based state input.

## Resources

The city environments from the paper 'manhattan32' and 'urban50' are included in the 'res' directory. Map information is formatted as PNG files with one pixel representing on grid world cell. The pixel color determines the type of cell according to

* red #ff0000 no-fly zone (NFZ)
* green #00ff00 buildings blocking wireless links (UAVs can fly over)
* blue #0000ff start and landing zone
* yellow #ffff00 buildings blocking wireless links + NFZ (UAVs can not fly over)

If you would like to create a new map, you can use any tool to design a PNG with the same pixel dimensions as the desired map and the above color codes.

The shadowing maps, defining for each position and each IoT device whether there is a line-of-sight (LoS) or non-line-of-sight (NLoS) connection, are computed automatically the first time a new map is used for training and then saved to the 'res' directory as an NPY file.


## Reference

If using this code for research purposes, please cite:

[1] H. Bayerlein, M. Theile, M. Caccamo, and D. Gesbert, “Multi-UAV path planning for wireless data harvesting with deep reinforcement learning," arXiv:2010.12461 [cs.MA], 2020. 

```
@article{Bayerlein2020,
        author  = {Harald Bayerlein and Mirco Theile and Marco Caccamo and David Gesbert},
        title   = {Multi-{UAV} Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning},
        journal = {arXiv:2010.12461 [cs.MA]},
        year    = {2020},
        url     = {https://arxiv.org/abs/2010.12461}
}
```

Or for the shorter single-UAV conference version:

[2] H. Bayerlein, M. Theile, M. Caccamo, and D. Gesbert, “UAV path planning for wireless data harvesting: A deep reinforcement learning approach,” in IEEE Global Communications Conference (GLOBECOM), 2020.

```
@inproceedings{Bayerlein2020short,
  author={Harald Bayerlein and Mirco Theile and Marco Caccamo and David Gesbert},
  title={{UAV} Path Planning for Wireless Data Harvesting: A Deep Reinforcement Learning Approach}, 
  booktitle={IEEE Global Communications Conference (GLOBECOM)}, 
  year={2020}
}
```


## License 

This code is under a BSD license.