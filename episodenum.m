function y = episodenum(u)
%% EPISODENUM Get the next episode number based on existing .mat files
% This function calculates the next episode number by counting existing .mat files
% in the current directory and returning the count plus one.
%
% Input:
%   u - Dummy input (not used, maintained for compatibility with Simulink)
%
% Output:
%   y - The next episode number (integer)
%
% Example:
%   next_episode = episodenum([]);
%
% Note:
%   This function is typically used in reinforcement learning workflows to maintain
%   a sequential numbering of training episodes when saving agent data.

% Get the current working directory

% Get the current working directory
source_dir = pwd;

% Get a list of all .mat files in the current directory
d = dir([source_dir, '\*.mat']);

% Calculate the next episode number (current count + 1)
y = length(d) + 1;
