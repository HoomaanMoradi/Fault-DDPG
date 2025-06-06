%% Two-Link Robot Arm Control using DDPG
% This script implements a Deep Deterministic Policy Gradient (DDPG) agent to control
% a two-link robot arm in a Simulink environment. The agent learns to apply
% appropriate torques to the robot joints to achieve desired control objectives.
%
% Author: [Your Name]
% Date: [Date]
% Copyright: [Your Organization]

% Clear workspace, command window, close all figures, and delete .mat files
clear
clc
close all
delete *.mat

% Load and open the Simulink model
mdl = 'twolinkenv';
open_system(mdl)

% Define the path to the RL Agent block in the Simulink model
agentBlk = [mdl '/RL Agent'];

% Define observation specifications for the RL environment
% Observations include joint angles and velocities (normalized between -1 and 1)
% and their derivatives (unbounded)
obsInfo = rlNumericSpec([6 1],...
    'LowerLimit',[-1 -1 -inf -1 -1 -inf]',...
    'UpperLimit',[1 1 inf 1 1 inf]');
obsInfo.Name = 'observations';


% Define action specifications (torque applied to each joint)
% Torque limits are set to ±15 Nm for each joint
actInfo = rlNumericSpec([2 1],...
    'LowerLimit',[-15;-15],...
    'UpperLimit',[15;15]);
actInfo.Name = 'torque';


% Create the reinforcement learning environment
env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo);
% Set the reset function to update the episode number
env.ResetFcn = @(in)setVariable(in,'epinum',episodenum(1),'Workspace',mdl);

% Set simulation parameters
Ts = 0.02;  % Sample time (seconds)
Tf = 25;    % Simulation time for each episode (seconds)

% Set random seed for reproducibility
rng(0)


% Define the critic network architecture
% State path processes the observation input
statePath = [
    imageInputLayer([6 1 1], 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(128, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(200, 'Name', 'CriticStateFC2')];

% Action path processes the action input
actionPath = [
    imageInputLayer([2 1 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(200, 'Name', 'CriticActionFC1', 'BiasLearnRateFactor', 0)];

% Common path combines state and action paths and produces Q-value estimate
commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')];

% Assemble the critic network
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
    
% Connect the state and action paths to the common path
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% Configure critic representation options
criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1,'UseDevice',"gpu");

% Create critic representation
critic = rlRepresentation(criticNetwork,criticOptions,'Observation',{'observation'},obsInfo,'Action',{'action'},actInfo);

% Define the actor network architecture
% The actor takes observations as input and outputs actions (torques)
actorNetwork = [
    imageInputLayer([6 1 1], 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(128, 'Name', 'ActorFC1')
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(200, 'Name', 'ActorFC2')
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(2, 'Name', 'ActorFC3')
    tanhLayer('Name', 'ActorTanh1')  % Output between -1 and 1
    scalingLayer('Name','ActorScaling','Scale',reshape([15;15],[1,1,2]))];  % Scale to torque limits

% Configure actor representation options
actorOptions = rlRepresentationOptions('LearnRate',5e-04,'GradientThreshold',1,'UseDevice',"gpu");

% Create actor representation
actor = rlRepresentation(actorNetwork,actorOptions,'Observation',{'observation'},obsInfo,'Action',{'ActorScaling'},actInfo);

% Configure DDPG agent options
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...  % For target network updates
    'ExperienceBufferLength',1e6,...  % Size of experience buffer
    'MiniBatchSize',128);  % Batch size for training
% Configure exploration noise
agentOptions.NoiseOptions.Variance = [0.4;0.4];  % Initial exploration noise
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;  % Noise decay rate

% Create DDPG agent with specified actor, critic, and options
agent = rlDDPGAgent(actor,critic,agentOptions);

% Set up training parameters
maxepisodes = 2000;  % Maximum number of training episodes
maxsteps = ceil(Tf/Ts);  % Maximum steps per episode

% Configure training options
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',5,...  % Window size for averaging episode rewards
    'Verbose',false,...  % Disable command line display
    'Plots','training-progress',...  % Show training progress
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',0,...  % Stop if average reward reaches 0
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',-100000,...  % Save agent if episode reward exceeds this value
     'SaveAgentDirectory', pwd);  % Directory to save trained agents

% Train the DDPG agent
trainingStats = train(agent,env,trainingOptions);

