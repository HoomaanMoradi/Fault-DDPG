# Fault-Tolerant Control of Two-Link Robot Arm using DDPG

![MATLAB](https://img.shields.io/badge/MATLAB-R2021b%2B-blue)

This repository contains the implementation of a Deep Deterministic Policy Gradient (DDPG) agent for fault-tolerant control of a two-link robot arm in a Simulink environment. The agent learns to apply appropriate torques to the robot joints to achieve desired control objectives, even in the presence of faults.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training the Agent](#training-the-agent)

## Overview

This project implements a DDPG-based controller for a two-link robot arm. The agent interacts with a Simulink environment, learning optimal control policies through reinforcement learning. The implementation includes:

- Custom actor-critic neural network architectures
- Experience replay buffer for stable training
- Exploration noise with decay for better policy learning
- Automatic saving of trained agents
- Visualization of training progress

## Prerequisites

- MATLAB R2021b or later
- Deep Learning Toolbox
- Reinforcement Learning Toolbox
- Simulink
- Parallel Computing Toolbox (recommended for faster training)
- CUDA-capable GPU (recommended for faster training)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Fault-DDPG.git
   cd Fault-DDPG
   ```

2. Open MATLAB and navigate to the project directory.

## Project Structure

- `twolinkrun.m`: Main script to train the DDPG agent
- `episodenum.m`: Helper function to manage episode numbering
- `twolinkenv.slx`: Simulink model of the two-link robot arm environment
- `*.mat`: Saved agent checkpoints (generated during training)

## Usage

1. Open MATLAB and navigate to the project directory.
2. Open the Simulink model to inspect the environment:
   ```matlab
   open_system('twolinkenv.slx');
   ```
3. Run the main training script:
   ```matlab
   twolinkrun;
   ```

## Training the Agent

The training process can be customized by modifying the following parameters in `twolinkrun.m`:

- `maxepisodes`: Maximum number of training episodes (default: 2000)
- `Tf`: Duration of each episode in seconds (default: 25s)
- `Ts`: Sample time in seconds (default: 0.02s)
- Network architectures in the actor and critic definitions
- Training options like learning rates and batch sizes

During training, the script will display a training progress plot showing:
- Episode reward
- Average reward
- Episode Q-value
- Episode steps