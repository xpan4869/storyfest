% This script converts raw EDF file to a matlab file

clear all; clc;
addpath('../scripts/uzh-edf-converter-e4841bf');


% Set Directories
dirs.ETraw = '../data/pupil/1_raw/recall';
dirs.ETfile = '../data/pupil/2_mat';

subjects = [1035:1042];
nSub = length(subjects);

% Compile data
for s = 1:nSub

    sub = num2str(subjects(s));
    fprintf('Running Subject %s \n', sub);

    % Load data
    ETpath = fullfile(dirs.ETraw, sprintf('%s_storyfest_recall.EDF', sub));

    ETraw = Edf2Mat(ETpath); 

  
    % Check to make sure sampling rate is 500 samples per second
    samp_interval = ETraw.Samples.time(2) - ETraw.Samples.time(1);
    if samp_interval == 2
        fprintf('Sampling Rate = 500Hz \n');
    else
        error('Error: Sub %s, samp_interval = %i',sub, samp_interval);
    end

    % Save 'Samples' and 'Events'
    Samples = ETraw.Samples;
    Events = ETraw.Events;

    % Make output directory if it doesn't exist
    savepath = '../data/pupil/2_mat/recall';
    %savepath = fullfile(dirs.ETfile, sub);
    if ~exist(savepath), mkdir(savepath); end

    save(fullfile(savepath, sprintf('%s_recall_ET.mat', sub)), 'Samples', 'Events', '-v7.3')

end
